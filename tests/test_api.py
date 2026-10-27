"""API 层测试。

通过 FastAPI 的依赖覆盖（dependency_overrides）把真实的大模型与向量库
替换成测试替身，因此这些用例完全离线、秒级完成。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.container import get_ingestion_pipeline, get_rag_chain, get_vector_store
from app.main import app
from tests.fakes import FakeIngestionPipeline, FakeRagChain, FakeVectorStore

_FILES = {"files": ("测试文档.md", b"# hello\ncontent", "text/markdown")}


def _make_client(store=None, chain=None, pipeline=None) -> TestClient:
    app.dependency_overrides[get_vector_store] = lambda: store or FakeVectorStore()
    app.dependency_overrides[get_rag_chain] = lambda: chain or FakeRagChain()
    app.dependency_overrides[get_ingestion_pipeline] = (
        lambda: pipeline or FakeIngestionPipeline()
    )
    # 不使用 with，从而跳过 lifespan，避免真实初始化向量库
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


# ------------------------------------------------------------------ 系统
def test_health_reports_status():
    client = _make_client()
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["indexed_chunks"] == 7
    assert "llm_model" in payload


# ------------------------------------------------------------------ 问答
def test_chat_returns_answer_with_citations():
    client = _make_client()
    response = client.post("/api/chat", json={"question": "产品支持哪些格式？"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "这是测试答案 [1]"
    assert payload["used_context"] is True
    assert payload["citations"][0]["source"] == "测试文档.md"
    assert payload["citations"][0]["index"] == 1


def test_chat_rejects_empty_question():
    client = _make_client()
    response = client.post("/api/chat", json={"question": ""})

    assert response.status_code == 422, "空问题应由参数校验直接拒绝"


def test_chat_stream_emits_sse_frames():
    client = _make_client()

    with client.stream(
        "POST", "/api/chat/stream", json={"question": "你好", "session_id": "s1"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "data:" in body
    assert '"type": "citations"' in body
    assert "测试答案" in body
    assert body.rstrip().endswith("[DONE]")


def test_stream_error_does_not_expose_internal_details():
    class BrokenChain:
        def stream(self, *args, **kwargs):
            raise RuntimeError("private-provider-response")

    client = _make_client(chain=BrokenChain())
    response = client.post("/api/chat/stream", json={"question": "测试"})
    assert '"type": "error"' in response.text
    assert "private-provider-response" not in response.text
    assert "检查服务日志" in response.text
    assert response.text.rstrip().endswith("[DONE]")


def test_reset_session_returns_ok():
    client = _make_client()
    response = client.delete("/api/chat/session/s1")

    assert response.status_code == 200
    assert response.json()["session_id"] == "s1"


# ------------------------------------------------------------------ 文档
def test_list_documents():
    client = _make_client(store=FakeVectorStore(count=12, sources=[{"source": "a.md", "chunks": 12}]))
    response = client.get("/api/documents")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_documents"] == 1
    assert payload["total_chunks"] == 12
    assert payload["documents"][0]["source"] == "a.md"


def test_upload_document_success():
    pipeline = FakeIngestionPipeline(files=1, chunks=4)
    client = _make_client(pipeline=pipeline)

    response = client.post("/api/documents/upload", files=_FILES)

    assert response.status_code == 200
    assert response.json()["chunks"] == 4
    assert len(pipeline.calls) == 1, "入库管道应被调用一次"


def test_upload_rejects_unsupported_extension():
    client = _make_client()
    response = client.post(
        "/api/documents/upload",
        files={"files": ("malware.exe", b"MZ", "application/octet-stream")},
    )

    assert response.status_code == 415


def test_upload_sanitises_path_traversal_filename():
    pipeline = FakeIngestionPipeline()
    client = _make_client(pipeline=pipeline)

    response = client.post(
        "/api/documents/upload",
        files={"files": ("../../etc/passwd.md", b"pwned", "text/markdown")},
    )

    assert response.status_code == 200
    saved_path = pipeline.calls[0][0]
    assert saved_path.name == "passwd.md", "目录部分必须被剥离"
    assert ".." not in str(saved_path)


def test_delete_missing_document_returns_404():
    client = _make_client(store=FakeVectorStore(deletable={}))
    response = client.delete("/api/documents/不存在.md")

    assert response.status_code == 404


def test_delete_existing_document():
    client = _make_client(store=FakeVectorStore(deletable={"a.md": 3}))
    response = client.delete("/api/documents/a.md")

    assert response.status_code == 200
    assert response.json()["deleted_chunks"] == 3
