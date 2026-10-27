"""文本分块。

分块策略是 RAG 效果的第一道分水岭，这里做了三件事：
1. **中文友好分隔符**：优先在段落 / 中文句号处切分，避免把一句话拦腰截断
2. **重叠（overlap）**：相邻块保留少量重叠，缓解「答案被切在边界」的问题
3. **稳定 chunk_id**：由 ``doc_id + 序号`` 组成，重新入库时覆盖旧块而非重复堆积
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

# 从粗到细：段落 -> 换行 -> 中文句末标点 -> 英文句末 -> 逗号 -> 空格 -> 硬切
DEFAULT_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "……",
    ". ",
    "! ",
    "? ",
    "; ",
    "，",
    ", ",
    " ",
    "",
]


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
        length_function=len,
        keep_separator=True,
        add_start_index=True,
    )


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """切分文档，并为每个块补上 ``chunk_index`` 与 ``chunk_id`` 元数据。"""
    if not documents:
        return []

    splitter = build_splitter(chunk_size, chunk_overlap)
    raw_chunks = splitter.split_documents(documents)

    counters: dict[str, int] = {}
    chunks: list[Document] = []
    for chunk in raw_chunks:
        content = chunk.page_content.strip()
        if len(content) < 10:  # 丢掉页眉页脚之类的碎片
            continue

        doc_id = str(chunk.metadata.get("doc_id", "unknown"))
        index = counters.get(doc_id, 0)
        counters[doc_id] = index + 1

        chunk.page_content = content
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = f"{doc_id}-{index:04d}"
        chunks.append(chunk)

    return chunks
