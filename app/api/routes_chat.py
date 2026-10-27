"""问答相关接口：普通问答、流式问答、重置会话。"""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest, ChatResponse
from app.container import get_rag_chain
from app.core.rag_chain import RagChain
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["问答"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 关闭 Nginx 缓冲，保证逐字输出
}


@router.post("", response_model=ChatResponse, summary="知识库问答（非流式）")
def chat(payload: ChatRequest, chain: RagChain = Depends(get_rag_chain)) -> ChatResponse:
    result = chain.answer(
        payload.question, session_id=payload.session_id, top_k=payload.top_k
    )
    return ChatResponse(**result.to_dict())


@router.post("/stream", summary="知识库问答（SSE 流式）")
def chat_stream(
    payload: ChatRequest, chain: RagChain = Depends(get_rag_chain)
) -> StreamingResponse:
    """以 Server-Sent Events 逐段推送：先给引用，再逐 token 给答案。"""

    def event_stream() -> Iterator[str]:
        try:
            for event in chain.stream(
                payload.question, session_id=payload.session_id, top_k=payload.top_k
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001 - 流式场景必须把异常也推给前端
            logger.exception("流式问答失败")
            error = {"type": "error", "data": f"{type(exc).__name__}: {exc}"}
            yield f"data: {json.dumps(error, ensure_ascii=False)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS
    )


@router.delete("/session/{session_id}", summary="清空指定会话的上下文")
def reset_session(session_id: str, chain: RagChain = Depends(get_rag_chain)) -> dict:
    chain.clear_session(session_id)
    return {"status": "ok", "session_id": session_id}
