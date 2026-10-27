"""入库管道测试：加载 -> 分块 -> 写入的端到端行为（不依赖网络）。"""

from __future__ import annotations

import pytest

from app.core.vectorstore import VectorStore
from app.ingestion.loader import UnsupportedFileTypeError, load_file
from app.ingestion.pipeline import IngestionPipeline
from tests.fakes import FakeEmbeddings


def _pipeline(tmp_chroma, name: str = "pipe") -> tuple[IngestionPipeline, VectorStore]:
    store = VectorStore(embeddings=FakeEmbeddings(), persist_dir=tmp_chroma, collection_name=name)
    return IngestionPipeline(store), store


def _write_docs(directory) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "产品说明.md").write_text(
        "# 产品说明\n\n" + "云雀智答是一款企业知识库问答产品。" * 20, encoding="utf-8"
    )
    (directory / "notes.txt").write_text(
        "这是一份纯文本笔记，包含若干说明性内容。" * 20, encoding="utf-8"
    )
    # 不支持的格式，应被扫描阶段忽略
    (directory / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")


def test_ingest_directory_writes_documents(tmp_path, tmp_chroma):
    pipeline, store = _pipeline(tmp_chroma)
    docs_dir = tmp_path / "docs"
    _write_docs(docs_dir)

    result = pipeline.ingest_directory(docs_dir)

    assert result.files == 2, "csv 应被忽略，只处理 md 与 txt"
    assert result.chunks > 0
    assert store.count == result.chunks
    assert not result.errors

    sources = {item["source"] for item in store.list_sources()}
    assert sources == {"产品说明.md", "notes.txt"}


def test_reingest_is_idempotent(tmp_path, tmp_chroma):
    """同一批文件重复入库，不应产生重复文本块。"""
    pipeline, store = _pipeline(tmp_chroma, "idempotent")
    docs_dir = tmp_path / "docs"
    _write_docs(docs_dir)

    pipeline.ingest_directory(docs_dir)
    first_count = store.count

    pipeline.ingest_directory(docs_dir)

    assert store.count == first_count


def test_ingest_missing_file_records_error(tmp_path, tmp_chroma):
    pipeline, _ = _pipeline(tmp_chroma, "errors")
    result = pipeline.ingest_files([tmp_path / "不存在.md"])

    assert result.files == 0
    assert result.errors, "缺失文件应被记录为错误而不是抛异常中断"


def test_empty_file_is_skipped(tmp_path, tmp_chroma):
    pipeline, store = _pipeline(tmp_chroma, "empty")
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")

    result = pipeline.ingest_files([empty])

    assert result.files == 0
    assert result.skipped
    assert store.count == 0


def test_load_file_rejects_unsupported_extension(tmp_path):
    target = tmp_path / "report.xlsx"
    target.write_bytes(b"fake")

    with pytest.raises(UnsupportedFileTypeError):
        load_file(target)


def test_load_file_decodes_non_utf8_text(tmp_path):
    """GBK 编码的中文文档不应乱码。"""
    target = tmp_path / "gbk.txt"
    target.write_bytes("中文编码测试内容，需要足够长以便通过过滤。".encode("gb18030"))

    documents = load_file(target)

    assert len(documents) == 1
    assert "中文编码测试内容" in documents[0].page_content
    assert documents[0].metadata["source"] == "gbk.txt"
    assert documents[0].metadata["file_type"] == "txt"
    assert "doc_id" in documents[0].metadata
