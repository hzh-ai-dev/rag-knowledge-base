"""分块逻辑测试。"""

from __future__ import annotations

from langchain_core.documents import Document

from app.ingestion.splitter import split_documents


def _doc(text: str, doc_id: str = "abc123", source: str = "test.md") -> Document:
    return Document(page_content=text, metadata={"doc_id": doc_id, "source": source})


def test_short_document_kept_as_single_chunk():
    chunks = split_documents([_doc("这是一段很短的中文文本，用于测试。")], 500, 50)
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_id"] == "abc123-0000"
    assert chunks[0].metadata["chunk_index"] == 0


def test_long_chinese_document_is_split():
    text = "。".join(f"这是第{i}句话，用于验证分块逻辑是否正常工作" for i in range(60)) + "。"
    chunks = split_documents([_doc(text)], 200, 40)

    assert len(chunks) > 1
    # RecursiveCharacterTextSplitter 保证除极端情况外不超过 chunk_size
    assert all(len(c.page_content) <= 260 for c in chunks)


def test_chunk_ids_unique_and_deterministic():
    text = "。".join(f"第{i}句内容用于测试稳定标识" for i in range(80))
    first = split_documents([_doc(text)], 150, 30)
    second = split_documents([_doc(text)], 150, 30)

    ids_first = [c.metadata["chunk_id"] for c in first]
    ids_second = [c.metadata["chunk_id"] for c in second]

    assert len(ids_first) == len(set(ids_first)), "chunk_id 必须唯一"
    assert ids_first == ids_second, "同样的输入必须产生同样的 chunk_id"


def test_multiple_documents_get_independent_indexes():
    docs = [_doc("甲" * 400, doc_id="aaa"), _doc("乙" * 400, doc_id="bbb")]
    chunks = split_documents(docs, 150, 30)

    ids = {c.metadata["doc_id"] for c in chunks}
    assert ids == {"aaa", "bbb"}
    assert all(c.metadata["chunk_id"].startswith(c.metadata["doc_id"]) for c in chunks)


def test_fragments_shorter_than_threshold_are_dropped():
    assert split_documents([_doc("短"), _doc("也短")]) == []
