"""依赖容器：集中管理各类单例，供 FastAPI 依赖注入使用。

好处：
- 对象只创建一次（向量库连接、BM25 索引缓存等代价较高）
- 测试时可以用 ``app.dependency_overrides`` 轻松替换成假实现，无需真实调用大模型
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.core.embeddings import get_embedding_function
from app.core.memory import ConversationMemory
from app.core.rag_chain import RagChain
from app.core.retriever import HybridRetriever
from app.core.vectorstore import VectorStore
from app.ingestion.pipeline import IngestionPipeline


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore(
        embeddings=get_embedding_function(),
        persist_dir=settings.chroma_dir,
        collection_name=settings.collection_name,
    )


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever(get_vector_store())


@lru_cache(maxsize=1)
def get_memory() -> ConversationMemory:
    return ConversationMemory()


@lru_cache(maxsize=1)
def get_rag_chain() -> RagChain:
    return RagChain(retriever=get_retriever(), memory=get_memory())


@lru_cache(maxsize=1)
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(store=get_vector_store())
