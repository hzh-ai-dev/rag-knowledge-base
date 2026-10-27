"""向量模型工厂。

支持两种后端：
- ``openai``：调用任意 OpenAI 兼容的 /embeddings 接口（OpenAI、通义、智谱等）
- ``local`` ：使用 sentence-transformers 本地模型，零 API 成本、可离线

对外只暴露一个 ``get_embedding_function()``，调用方无需关心后端差异，
这也是把「配置」和「实现」解耦的典型做法。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_function() -> Embeddings:
    """按配置返回向量模型单例（进程内缓存，避免重复初始化）。"""
    provider = settings.embedding_provider.lower()
    if provider == "local":
        logger.info("使用本地向量模型: %s", settings.local_embedding_model)
        return _build_local_embeddings()
    logger.info("使用远程向量模型: %s @ %s", settings.embedding_model, settings.openai_base_url)
    return _build_openai_embeddings()


def _build_openai_embeddings() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        chunk_size=settings.embedding_batch_size,
        # 关键：部分国产厂商的 embedding 接口不接受 tiktoken 分词预处理，
        # 关掉后可避免 "context length" 相关的兼容性报错。
        check_embedding_ctx_length=False,
    )


def _build_local_embeddings() -> Embeddings:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的友好提示
        raise RuntimeError(
            "本地向量模型需要额外依赖，请先执行：\n"
            "  pip install sentence-transformers langchain-community"
        ) from exc

    return HuggingFaceEmbeddings(
        model_name=settings.local_embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )
