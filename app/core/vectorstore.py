"""向量库封装（ChromaDB）。

为什么单独抽一层：
- 把「存储引擎」和「业务逻辑」解耦，将来换成 Qdrant / Milvus 只需重写本文件
- 统一处理 embedding、元数据清洗、相似度换算、按来源删除等重复逻辑
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.utils.logging import get_logger

logger = get_logger(__name__)

# Chroma 的元数据只接受 str / int / float / bool
_SCALAR_TYPES = (str, int, float, bool)


@dataclass
class RetrievedChunk:
    """一条检索结果。``score`` 越大越相关（已统一为相似度）。"""

    chunk_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    @property
    def page(self) -> int | None:
        page = self.metadata.get("page")
        return int(page) if page is not None else None


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """过滤掉 Chroma 不支持的字段类型，避免写入时报错。"""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, _SCALAR_TYPES):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


class VectorStore:
    """ChromaDB 的轻量封装，负责写入 / 查询 / 统计 / 删除。"""

    def __init__(
        self,
        embeddings: Embeddings,
        persist_dir: str | Path,
        collection_name: str,
    ) -> None:
        self._embeddings = embeddings
        self._version = 0  # 写入版本号，供 BM25 索引判断是否需要重建
        self._persist_dir = str(persist_dir)
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 余弦距离，配合归一化向量效果稳定
        )

    # ------------------------------------------------------------------ 写
    def add_documents(self, documents: list[Document]) -> int:
        """批量写入。使用 upsert，同一 chunk_id 重复写入会覆盖（幂等更新）。"""
        if not documents:
            return 0

        texts = [d.page_content for d in documents]
        vectors = self._embeddings.embed_documents(texts)
        ids = [str(d.metadata["chunk_id"]) for d in documents]
        metadatas = [_sanitize_metadata(d.metadata) for d in documents]

        self._collection.upsert(
            ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas
        )
        self._version += 1
        logger.info("已写入 %d 个文本块", len(documents))
        return len(documents)

    def delete_by_source(self, source: str) -> int:
        """删除某个文件的所有文本块（重新入库前先清旧数据）。"""
        existing = self._collection.get(where={"source": source}, include=[])
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
            self._version += 1
            logger.info("已删除来源 %s 的 %d 个文本块", source, len(ids))
        return len(ids)

    def reset(self) -> None:
        """清空整个集合。"""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name, metadata={"hnsw:space": "cosine"}
        )
        self._version += 1
        logger.warning("向量集合已清空")

    # ------------------------------------------------------------------ 读
    def query(
        self,
        question: str,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """向量相似度检索。"""
        if self.count == 0:
            return []

        query_vector = self._embeddings.embed_query(question)
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, max(self.count, 1)),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for cid, content, meta, dist in zip(ids, docs, metas, dists, strict=False):
            # cosine 距离 -> 相似度，范围大致落在 [0, 1]
            similarity = 1.0 - float(dist)
            chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    content=content,
                    metadata=dict(meta or {}),
                    score=similarity,
                )
            )
        return chunks

    def all_documents(self) -> list[RetrievedChunk]:
        """取出全部文本块，供 BM25 / 关键词检索构建索引使用。"""
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        return [
            RetrievedChunk(chunk_id=cid, content=content, metadata=dict(meta or {}))
            for cid, content, meta in zip(ids, docs, metas, strict=False)
        ]

    @property
    def count(self) -> int:
        return self._collection.count()

    @property
    def version(self) -> int:
        """单调递增的写入版本号，用于让下游缓存（如 BM25 索引）失效。"""
        return self._version

    def list_sources(self) -> list[dict[str, Any]]:
        """列出已入库的文件及各自的文本块数量。"""
        result = self._collection.get(include=["metadatas"])
        counter: dict[str, int] = {}
        for meta in result.get("metadatas") or []:
            source = str((meta or {}).get("source", "unknown"))
            counter[source] = counter.get(source, 0) + 1
        return [
            {"source": src, "chunks": n}
            for src, n in sorted(counter.items(), key=lambda kv: kv[0])
        ]
