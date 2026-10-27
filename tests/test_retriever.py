"""检索层测试：分词、RRF 融合、向量库读写。"""

from __future__ import annotations

from langchain_core.documents import Document

from app.core.retriever import rrf_fuse, tokenize
from app.core.vectorstore import RetrievedChunk, VectorStore
from tests.fakes import FakeEmbeddings


def _chunk(chunk_id: str, content: str | None = None, source: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content or f"内容 {chunk_id}",
        metadata={"source": source or f"{chunk_id}.md"},
        score=0.0,
    )


# ------------------------------------------------------------------ 分词
def test_tokenize_handles_mixed_language():
    tokens = tokenize("RAG 检索增强 v3.2 效果不错")

    assert "rag" in tokens          # 英文转小写
    assert "检" in tokens            # 中文按单字
    assert "v3" in tokens            # 字母数字整体保留
    assert "。" not in tokens        # 标点被丢弃


def test_tokenize_empty_string():
    assert tokenize("   ") == []


# ------------------------------------------------------------------ RRF
def test_rrf_fuse_prefers_items_ranked_high_in_both_lists():
    vector_hits = [_chunk("x"), _chunk("y")]
    keyword_hits = [_chunk("y"), _chunk("z")]

    fused = rrf_fuse([vector_hits, keyword_hits], k=60)

    assert fused[0].chunk_id == "y", "两路都靠前的文档应排在首位"
    assert {c.chunk_id for c in fused} == {"x", "y", "z"}
    scores = [c.score for c in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fuse_weights_shift_ranking():
    # 两路结果必须互不重叠，才能真正验证「权重」对排名的影响：
    # 若存在两路都命中的文档，它会同时累加两路的 1/(k+rank)，
    # 从而掩盖权重差异（这种共识行为由下一个用例单独覆盖）。
    vector_hits = [_chunk("vector_1"), _chunk("vector_2")]
    keyword_hits = [_chunk("keyword_1"), _chunk("keyword_2")]

    vector_heavy = rrf_fuse([vector_hits, keyword_hits], k=60, weights=[3.0, 1.0])
    keyword_heavy = rrf_fuse([vector_hits, keyword_hits], k=60, weights=[1.0, 3.0])

    assert vector_heavy[0].chunk_id == "vector_1", "权重偏向向量时，向量榜首应排第一"
    assert keyword_heavy[0].chunk_id == "keyword_1", "权重偏向关键词时，关键词榜首应排第一"


def test_rrf_fuse_rewards_documents_hit_by_both_retrievers():
    """两路都命中的文档会累加两路分数，因此排名应上升（共识即高分）。"""
    vector_hits = [_chunk("vector_1"), _chunk("shared")]
    keyword_hits = [_chunk("keyword_1"), _chunk("shared")]

    fused = rrf_fuse([vector_hits, keyword_hits], k=60)

    assert fused[0].chunk_id == "shared", "两路都命中的文档应排在两路各自的第一名之前"


def test_rrf_fuse_handles_single_list():
    fused = rrf_fuse([[_chunk("a"), _chunk("b")]], k=60)
    assert [c.chunk_id for c in fused] == ["a", "b"]


# ------------------------------------------------------------------ 向量库
def _store(tmp_chroma, name: str = "unit") -> VectorStore:
    return VectorStore(embeddings=FakeEmbeddings(), persist_dir=tmp_chroma, collection_name=name)


def test_vector_store_add_query_and_delete(tmp_chroma):
    store = _store(tmp_chroma)
    documents = [
        Document(
            page_content="云雀智答支持 PDF、Word、Markdown 与纯文本等文档格式。",
            metadata={"chunk_id": "d1-0000", "source": "格式说明.md", "doc_id": "d1"},
        ),
        Document(
            page_content="私有化部署按年授权，起步价为十八万元，包含三个实例。",
            metadata={"chunk_id": "d2-0000", "source": "报价单.md", "doc_id": "d2"},
        ),
    ]

    assert store.add_documents(documents) == 2
    assert store.count == 2

    hits = store.query("支持哪些文档格式？", top_k=2)
    assert hits, "应当检索到结果"
    assert hits[0].source == "格式说明.md"
    assert 0.0 <= hits[0].score <= 1.0 + 1e-6

    assert store.delete_by_source("格式说明.md") == 1
    assert store.count == 1
    assert store.delete_by_source("不存在的.md") == 0


def test_vector_store_list_sources(tmp_chroma):
    store = _store(tmp_chroma, "sources")
    store.add_documents(
        [
            Document(page_content="第一部分内容，足够长以便通过过滤。" * 3,
                     metadata={"chunk_id": "s-0000", "source": "甲.md", "doc_id": "s"}),
            Document(page_content="第二部分内容，同样足够长以便通过过滤。" * 3,
                     metadata={"chunk_id": "s-0001", "source": "甲.md", "doc_id": "s"}),
            Document(page_content="第三个文件的内容，长度也足够用于测试。" * 3,
                     metadata={"chunk_id": "t-0000", "source": "乙.md", "doc_id": "t"}),
        ]
    )

    sources = {item["source"]: item["chunks"] for item in store.list_sources()}
    assert sources == {"甲.md": 2, "乙.md": 1}


def test_vector_store_version_changes_on_write(tmp_chroma):
    store = _store(tmp_chroma, "version")
    initial = store.version

    store.add_documents(
        [
            Document(
                page_content="一段用于触发写入的内容，需要足够长。" * 3,
                metadata={"chunk_id": "v-0000", "source": "v.md", "doc_id": "v"},
            )
        ]
    )
    assert store.version > initial


def test_vector_store_query_on_empty_collection(tmp_chroma):
    store = _store(tmp_chroma, "empty")
    assert store.query("任意问题", top_k=3) == []


def test_metadata_is_sanitised_for_chroma(tmp_chroma):
    """Chroma 只接受标量元数据，嵌套结构应被转成字符串而不是报错。"""
    store = _store(tmp_chroma, "meta")
    store.add_documents(
        [
            Document(
                page_content="带复杂元数据的文档内容，长度需要超过过滤阈值。" * 2,
                metadata={
                    "chunk_id": "m-0000",
                    "source": "m.md",
                    "doc_id": "m",
                    "tags": ["a", "b"],          # 列表 -> 字符串
                    "verified": True,            # bool 保留
                    "empty_field": None,         # None 丢弃
                },
            )
        ]
    )
    assert store.count == 1
