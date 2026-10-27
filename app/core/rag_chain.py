"""RAG 问答链：检索 -> 拼装上下文 -> 生成带引用的回答。

把「检索」和「生成」显式拆开写，而不是用一行 ``RetrievalQA.from_chain_type``，
原因是真实业务里这两步经常要各自调参、各自埋点、各自换实现；
显式编排让每个环节都可观测、可测试、可替换。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_llm
from app.core.memory import ConversationMemory
from app.core.retriever import HybridRetriever
from app.core.vectorstore import RetrievedChunk
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 提示词是整个 RAG 的「大脑」，规则写得越具体，模型越不容易跑偏
SYSTEM_PROMPT_TEMPLATE = """你是一位严谨的知识库问答助手。请严格遵守以下规则：

1. 只能依据下方【参考资料】作答，不要使用你的先验知识，也不要凭空推测。
2. 在使用了某条资料的句子末尾标注引用编号，格式为 [1]、[2]，可并列如 [1][3]。
3. 若【参考资料】不足以回答问题，直接回答「根据已有资料无法回答该问题」，绝不编造。
4. 使用简体中文回答，表达清晰；涉及步骤或并列要点时可以分点说明。
5. 保持简洁，只回答被问到的问题，不要复述无关资料。

【参考资料】
{context}
"""

NO_CONTEXT_ANSWER = "根据已有资料无法回答该问题。请先上传相关资料，或换一个与资料库内容相关的问题。"


@dataclass
class Citation:
    """一条引用来源，前端据此展示「答案出自哪里」。"""

    index: int
    source: str
    page: int | None
    score: float
    snippet: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "source": self.source,
            "page": self.page,
            "score": round(self.score, 4),
            "snippet": self.snippet,
        }


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    session_id: str = "default"
    used_context: bool = False

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "session_id": self.session_id,
            "used_context": self.used_context,
        }


def build_context(chunks: list[RetrievedChunk]) -> str:
    """把检索结果拼成带编号的参考资料块（编号与 Citation.index 一一对应）。"""
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        location = chunk.source
        if chunk.page is not None:
            location += f"，第 {chunk.page} 页"
        blocks.append(f"[{index}] （来源：{location}）\n{chunk.content.strip()}")
    return "\n\n".join(blocks)


def _to_citations(chunks: list[RetrievedChunk], snippet_len: int = 160) -> list[Citation]:
    citations: list[Citation] = []
    for index, chunk in enumerate(chunks, start=1):
        snippet = " ".join(chunk.content.split())
        if len(snippet) > snippet_len:
            snippet = snippet[:snippet_len] + "…"
        citations.append(
            Citation(
                index=index,
                source=chunk.source,
                page=chunk.page,
                score=chunk.score,
                snippet=snippet,
            )
        )
    return citations


class RagChain:
    """一次问答的完整编排。"""

    def __init__(
        self,
        retriever: HybridRetriever,
        memory: ConversationMemory | None = None,
    ) -> None:
        self._retriever = retriever
        self._memory = memory or ConversationMemory()

    # ------------------------------------------------------------------ 内部
    def _prepare(self, question: str, top_k: int | None = None) -> tuple[list[RetrievedChunk], str]:
        chunks = self._retriever.retrieve(question, top_k=top_k)
        context = build_context(chunks)
        return chunks, context

    def _build_messages(self, question: str, context: str, session_id: str) -> list:
        system = SystemMessage(
            content=SYSTEM_PROMPT_TEMPLATE.format(context=context or "（无）")
        )
        history = self._memory.history(session_id)
        return [system, *history, HumanMessage(content=question)]

    # ------------------------------------------------------------------ 对外
    def answer(
        self, question: str, session_id: str = "default", top_k: int | None = None
    ) -> AnswerResult:
        """非流式问答。"""
        chunks, context = self._prepare(question, top_k=top_k)
        self._memory.add_user(session_id, question)

        if not chunks:
            self._memory.add_assistant(session_id, NO_CONTEXT_ANSWER)
            return AnswerResult(
                answer=NO_CONTEXT_ANSWER,
                citations=[],
                session_id=session_id,
                used_context=False,
            )

        messages = self._build_messages(question, context, session_id)
        response = get_llm().invoke(messages)
        text = _message_text(response)
        self._memory.add_assistant(session_id, text)

        return AnswerResult(
            answer=text,
            citations=_to_citations(chunks),
            session_id=session_id,
            used_context=True,
        )

    def stream(
        self, question: str, session_id: str = "default", top_k: int | None = None
    ) -> Iterator[dict]:
        """流式问答。依次产出 citations -> token* -> done 事件（供 SSE 转发）。"""
        chunks, context = self._prepare(question, top_k=top_k)
        self._memory.add_user(session_id, question)

        citations = _to_citations(chunks)
        yield {
            "type": "citations",
            "data": [c.to_dict() for c in citations],
        }

        if not chunks:
            self._memory.add_assistant(session_id, NO_CONTEXT_ANSWER)
            yield {"type": "token", "data": NO_CONTEXT_ANSWER}
            yield {"type": "done", "data": {"used_context": False}}
            return

        messages = self._build_messages(question, context, session_id)
        collected: list[str] = []
        for piece in get_llm(streaming=True).stream(messages):
            text = _message_text(piece)
            if text:
                collected.append(text)
                yield {"type": "token", "data": text}

        full_answer = "".join(collected)
        self._memory.add_assistant(session_id, full_answer)
        yield {"type": "done", "data": {"used_context": True}}

    def clear_session(self, session_id: str) -> None:
        self._memory.clear(session_id)


def _message_text(message) -> str:
    """兼容不同模型返回的 content 结构（str 或分段列表）。"""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content or "")
