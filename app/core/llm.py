"""大模型工厂（OpenAI 兼容接口）。

只要厂商提供 OpenAI 兼容的 ``/chat/completions``，这里都能直接对接：
DeepSeek、通义千问、智谱 GLM、Moonshot、OpenAI 官方等。
切换模型只需改 ``.env``，代码零改动。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import settings


@lru_cache(maxsize=2)
def get_llm(streaming: bool = False) -> ChatOpenAI:
    """返回 ChatModel 单例。streaming=True 用于 SSE 流式输出场景。"""
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.request_timeout,
        max_retries=2,
        streaming=streaming,
    )
