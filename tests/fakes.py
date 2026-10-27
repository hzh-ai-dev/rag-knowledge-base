"""测试用替身（Fake）实现。

核心价值：让单测与集成测试完全不依赖网络与大模型，跑得快、可重复、不烧钱。
"""

from __future__ import annotations

import hashlib

import numpy as np
from langchain_core.embeddings import Embeddings

from app.core.rag_chain import AnswerResult, Citation
from app.core.retriever import tokenize


class FakeEmbeddings(Embeddings):
    """基于词袋哈希的确定性向量模型。

    同样的文本必然得到同样的向量；词重合度越高，余弦相似度越大，
    因此可以真实地检验「检索是否能找回相关文本」这一行为，而不是随机通过。
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[index] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeVectorStore:
    """仅实现接口契约的极小替身，供 API 层测试使用。"""

    def __init__(
        self,
        count: int = 7,
        sources: list[dict] | None = None,
        deletable: dict[str, int] | None = None,
    ) -> None:
        self._count = count
        self._sources = sources or [{"source": "测试文档.md", "chunks": 7}]
        self._deletable = deletable or {"测试文档.md": 7}
        self.reset_called = False

    @property
    def count(self) -> int:
        return self._count

    @property
    def version(self) -> int:
        return 1

    def list_sources(self) -> list[dict]:
        return self._sources

    def delete_by_source(self, source: str) -> int:
        return self._deletable.get(source, 0)

    def reset(self) -> None:
        self.reset_called = True


class FakeRagChain:
    """返回固定答案的问答链替身。"""

    def answer(self, question: str, session_id: str = "default", top_k: int | None = None) -> AnswerResult:
        return AnswerResult(
            answer="这是测试答案 [1]",
            citations=[
                Citation(index=1, source="测试文档.md", page=2, score=0.88, snippet="片段内容")
            ],
            session_id=session_id,
            used_context=True,
        )

    def stream(self, question: str, session_id: str = "default", top_k: int | None = None):
        yield {
            "type": "citations",
            "data": [
                {"index": 1, "source": "测试文档.md", "page": 2, "score": 0.88, "snippet": "片段"}
            ],
        }
        yield {"type": "token", "data": "这是"}
        yield {"type": "token", "data": "测试答案"}
        yield {"type": "done", "data": {"used_context": True}}

    def clear_session(self, session_id: str) -> None:
        return None


class FakeIngestionPipeline:
    def __init__(self, files: int = 1, chunks: int = 5) -> None:
        self._files = files
        self._chunks = chunks
        self.calls: list[list] = []

    def ingest_files(self, paths):
        self.calls.append(list(paths))

        class _Result:
            files = self._files
            chunks = self._chunks
            skipped: list[str] = []
            errors: list[str] = []

            def to_dict(self):
                return {
                    "files": self.files,
                    "chunks": self.chunks,
                    "skipped": self.skipped,
                    "errors": self.errors,
                }

        return _Result()

    def ingest_directory(self, directory):
        return self.ingest_files([])
