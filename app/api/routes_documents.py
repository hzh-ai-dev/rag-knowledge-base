"""文档管理接口：上传入库、列表、删除、重建索引。"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    IngestResponse,
)
from app.config import settings
from app.container import get_ingestion_pipeline, get_vector_store
from app.core.vectorstore import VectorStore
from app.ingestion.pipeline import IngestionPipeline
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["文档管理"])

_UNSAFE_CHARS = re.compile(r"[^\w一-鿿.\- ]")


def _safe_filename(name: str | None) -> str:
    """清洗上传文件名，防止路径穿越（../../etc/passwd 之类）。"""
    if not name:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    # 只取最后一段，去掉目录部分
    name = Path(name).name
    name = _UNSAFE_CHARS.sub("_", name).strip()
    if not name or name.startswith("."):
        raise HTTPException(status_code=400, detail="非法文件名")
    return name


def _validate_extension(name: str) -> None:
    if Path(name).suffix.lower() not in settings.allowed_extensions:
        allowed = "、".join(settings.allowed_extensions)
        raise HTTPException(
            status_code=415, detail=f"不支持的文件类型，仅支持：{allowed}"
        )


@router.post("/upload", response_model=IngestResponse, summary="上传文档并入库")
async def upload_documents(
    files: list[UploadFile] = File(..., description="支持 PDF / MD / TXT / DOCX"),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="未收到任何文件")

    saved_paths: list[Path] = []
    for upload in files:
        filename = _safe_filename(upload.filename)
        _validate_extension(filename)

        content = await upload.read()
        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} 超过大小上限 {settings.max_upload_mb}MB",
            )

        target = Path(settings.upload_dir) / filename
        target.write_bytes(content)
        saved_paths.append(target)
        logger.info("已保存上传文件 %s (%d 字节)", filename, len(content))

    result = pipeline.ingest_files(saved_paths)
    return IngestResponse(**result.to_dict())


@router.get("", response_model=DocumentListResponse, summary="查看已入库文档")
def list_documents(store: VectorStore = Depends(get_vector_store)) -> DocumentListResponse:
    sources = store.list_sources()
    return DocumentListResponse(
        total_documents=len(sources),
        total_chunks=store.count,
        documents=[DocumentInfo(**item) for item in sources],
    )


@router.delete("/{source}", response_model=DeleteResponse, summary="删除某个文档")
def delete_document(
    source: str, store: VectorStore = Depends(get_vector_store)
) -> DeleteResponse:
    deleted = store.delete_by_source(source)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"未找到文档：{source}")

    # 同步删除磁盘上的上传副本，保持状态一致
    file_path = Path(settings.upload_dir) / Path(source).name
    if file_path.exists():
        file_path.unlink()

    return DeleteResponse(deleted_chunks=deleted, source=source)


@router.post("/reindex", response_model=IngestResponse, summary="重建索引（重新扫描上传目录）")
def reindex(pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)) -> IngestResponse:
    result = pipeline.ingest_directory(settings.upload_dir)
    return IngestResponse(**result.to_dict())


@router.delete("", summary="清空整个知识库")
def reset_documents(store: VectorStore = Depends(get_vector_store)) -> dict:
    store.reset()
    for path in Path(settings.upload_dir).glob("*"):
        if path.is_file():
            path.unlink()
    return {"status": "ok", "message": "知识库已清空"}
