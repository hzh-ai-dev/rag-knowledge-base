"""FastAPI 应用入口。

启动：
    uvicorn app.main:app --reload --port 8000
或：
    python -m app.main
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_chat import router as chat_router
from app.api.routes_documents import router as documents_router
from app.api.schemas import HealthResponse
from app.config import settings
from app.container import get_vector_store
from app.core.vectorstore import VectorStore
from app.utils.errors import public_error
from app.utils.logging import get_logger, setup_logging

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.debug)
    settings.ensure_dirs()
    logger.info("=" * 60)
    logger.info("%s v%s 启动", settings.app_name, settings.app_version)
    logger.info("LLM        : %s @ %s", settings.llm_model, settings.openai_base_url)
    logger.info("Embedding  : %s (%s)", settings.embedding_model, settings.embedding_provider)
    logger.info(
        "检索策略   : hybrid=%s, rerank=%s, top_k=%d",
        settings.enable_hybrid,
        settings.enable_rerank,
        settings.top_k,
    )
    logger.info("正在加载向量模型和知识库，首次下载模型可能较慢，请等待启动完成…")
    logger.info("已入库块数 : %d", get_vector_store().count)
    logger.info("=" * 60)
    yield
    logger.info("服务已停止")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "基于 FastAPI + LangChain 的检索增强生成（RAG）知识库问答服务。\n\n"
        "支持：多格式文档入库、混合检索（向量 + BM25）、引用溯源、流式输出、多轮对话。"
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 业务路由 ----
app.include_router(chat_router, prefix="/api")
app.include_router(documents_router, prefix="/api")


@app.get("/health", response_model=HealthResponse, tags=["系统"], summary="健康检查")
def health(store: VectorStore = Depends(get_vector_store)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        indexed_chunks=store.count,
        hybrid_search=settings.enable_hybrid,
        rerank=settings.enable_rerank,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("未处理异常: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": public_error(exc)},
    )


# ---- 前端静态页面 ----
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "index.html"))


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)
