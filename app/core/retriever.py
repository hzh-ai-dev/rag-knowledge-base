"""检索层：向量检索 + BM25 关键词检索 + RRF 融合 + 可选重排。

为什么要做混合检索（Hybrid Search）？
- 纯向量检索对「专有名词 / 型号 / 编号」这类词容易漏（语义相近但字面完全不同）
- 纯关键词检索又无法处理同义表达
- 两者用 RRF（Reciprocal Rank Fusion）融合，取长补短，是工业界常见做法

RRF 只依赖「排名」而非「分数」，因此不需要归一化两个检索器的分数量纲，
适用于融合不同评分尺度的检索结果。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.config import settings
from app.core.vectorstore import RetrievedChunk, VectorStore
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 中英文混合分词：连续英文/数字当一个词，中文按单字切
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    """轻量分词，不引入 jieba 等重型依赖也能在中英混排上拿到不错效果。"""
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


def rrf_fuse(
    result_lists: list[list[RetrievedChunk]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion：``score = Σ w_i / (k + rank_i)``。

    纯函数、无副作用、易测试 —— 建议单独写用例覆盖。
    """
    scores: dict[str, float] = {}
    by_id: dict[str, RetrievedChunk] = {}

    for list_index, results in enumerate(result_lists):
        weight = weights[list_index] if weights else 1.0
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + rank)
            by_id.setdefault(chunk.chunk_id, chunk)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    fused: list[RetrievedChunk] = []
    for chunk_id, score in ordered:
        chunk = by_id[chunk_id]
        chunk.score = score
        fused.append(chunk)
    return fused


def _matches_where(chunk: RetrievedChunk, where: dict | None) -> bool:
    if not where:
        return True
    return all(chunk.metadata.get(key) == value for key, value in where.items())


@dataclass
class _BM25Index:
    """BM25 索引缓存：仅当向量库版本变化时才重建（避免每次问答都全量重算）。"""

    version: int = -1
    chunks: list[RetrievedChunk] = None  # type: ignore[assignment]
    bm25: BM25Okapi | None = None

    def ensure(self, store: VectorStore) -> bool:
        if self.bm25 is not None and self.version == store.version:
            return True

        chunks = store.all_documents()
        if not chunks:
            self.bm25 = None
            self.chunks = []
            self.version = store.version
            return False

        corpus = [tokenize(chunk.content) for chunk in chunks]
        # BM25Okapi 对全空语料会异常，做一次兜底
        if not any(corpus):
            self.bm25 = None
            self.chunks = []
            self.version = store.version
            return False

        self.bm25 = BM25Okapi(corpus)
        self.chunks = chunks
        self.version = store.version
        logger.info("BM25 索引已重建，共 %d 个文本块", len(chunks))
        return True

    def search(self, query: str, top_k: int, where: dict | None = None) -> list[RetrievedChunk]:
        if self.bm25 is None or not self.chunks:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.chunks, scores, strict=False), key=lambda pair: pair[1], reverse=True
        )

        hits: list[RetrievedChunk] = []
        for chunk, score in ranked:
            if score <= 0:
                continue
            if not _matches_where(chunk, where):
                continue
            hits.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata=dict(chunk.metadata),
                    score=float(score),
                )
            )
            if len(hits) >= top_k:
                break
        return hits


class HybridRetriever:
    """统一的检索入口，屏蔽底层是向量、关键词还是混合策略。"""

    def __init__(self, store: VectorStore) -> None:
        self._store = store
        self._bm25 = _BM25Index()

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.top_k

        vector_hits = self._store.query(
            question, top_k=settings.vector_top_k, where=where
        )
        if not settings.enable_hybrid:
            return vector_hits[:top_k]

        keyword_hits: list[RetrievedChunk] = []
        if self._bm25.ensure(self._store):
            keyword_hits = self._bm25.search(question, settings.bm25_top_k, where)

        if not keyword_hits:
            return vector_hits[:top_k]

        fused = rrf_fuse(
            [vector_hits, keyword_hits],
            k=settings.rrf_k,
            weights=[1.0, 1.0],
        )

        if settings.enable_rerank:
            fused = self._lexical_rerank(question, fused)

        return fused[:top_k]

    # ------------------------------------------------------------------
    @staticmethod
    def _lexical_rerank(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """轻量重排：用问题与文本块的字面重合度对结果做二次排序。

        生产环境可替换为 bge-reranker / Cohere Rerank 等交叉编码器模型，
        这里的实现零额外依赖，作为「可插拔重排」的演示。
        替换点就在这一个方法内，不影响其它代码。
        """
        query_tokens = set(tokenize(question))
        if not query_tokens:
            return chunks

        max_rrf = max((c.score for c in chunks), default=1.0) or 1.0
        rescored: list[tuple[float, RetrievedChunk]] = []
        for chunk in chunks:
            chunk_tokens = set(tokenize(chunk.content))
            overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
            normalized_rrf = chunk.score / max_rrf
            # 7:3 融合，兼顾排名信息与字面重合
            blended = 0.7 * normalized_rrf + 0.3 * overlap
            rescored.append((blended, chunk))

        rescored.sort(key=lambda pair: pair[0], reverse=True)
        result = []
        for score, chunk in rescored:
            chunk.score = score
            result.append(chunk)
        return result
