"""全局配置：所有可调参数集中在这里，通过环境变量 / .env 覆盖。

设计原则：
- 单一配置源（Single Source of Truth），业务代码不再散落魔法数字
- 使用 pydantic-settings 做类型校验，配置写错在启动时就报错，而不是运行到一半才炸
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（app/config.py -> 上一级）
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置。字段名自动映射同名的环境变量（大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 应用 ----------
    app_name: str = "RAG Knowledge Base"
    app_version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # ---------- LLM（OpenAI 兼容接口）----------
    openai_api_key: str = "sk-placeholder"
    openai_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    request_timeout: int = 60

    # ---------- 向量模型 ----------
    embedding_provider: str = "openai"  # openai | local
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_batch_size: int = 64

    # ---------- 分块 ----------
    chunk_size: int = 700
    chunk_overlap: int = 120

    # ---------- 检索 ----------
    top_k: int = 6
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rerank_top_n: int = 6
    enable_hybrid: bool = True
    enable_rerank: bool = False
    rrf_k: int = 60  # RRF 融合平滑参数，经验值 60

    # ---------- 存储 ----------
    data_dir: Path = Field(default=BASE_DIR / "data")
    chroma_dir: Path = Field(default=BASE_DIR / "data" / "chroma")
    upload_dir: Path = Field(default=BASE_DIR / "data" / "uploads")
    collection_name: str = "knowledge_base"

    # ---------- 对话 ----------
    max_history_turns: int = 6  # 最多携带多少轮历史（1 轮 = 用户 + 助手）
    max_question_chars: int = 2000

    # ---------- 上传 ----------
    max_upload_mb: int = 32
    allowed_extensions: tuple[str, ...] = (".pdf", ".md", ".txt", ".docx")

    # ------------------------------------------------------------------
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        """启动时确保运行期目录存在。"""
        for d in (self.data_dir, self.chroma_dir, self.upload_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """带缓存的配置读取，全进程单例。"""
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
