"""API 请求 / 响应模型（Pydantic）。

用途有三：
1. 自动校验入参，非法请求在校验层就被挡下
2. 自动生成 OpenAPI / Swagger 文档（访问 /docs 即可）
3. 让前端对接有明确的字段契约
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# --------------------------------------------------------------------- 问答
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str = Field("default", max_length=128, description="会话 ID，用于多轮对话")
    top_k: int | None = Field(None, ge=1, le=20, description="覆盖默认召回数量")


class CitationModel(BaseModel):
    index: int = Field(..., description="引用编号，与答案中的 [n] 对应")
    source: str = Field(..., description="来源文件名")
    page: int | None = Field(None, description="页码（PDF）")
    score: float = Field(..., description="相关度分数")
    snippet: str = Field(..., description="来源片段摘要")


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationModel] = Field(default_factory=list)
    session_id: str
    used_context: bool = Field(..., description="是否检索到资料并据此作答")


# --------------------------------------------------------------------- 文档
class DocumentInfo(BaseModel):
    source: str
    chunks: int


class DocumentListResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: list[DocumentInfo] = Field(default_factory=list)


class IngestResponse(BaseModel):
    files: int
    chunks: int
    skipped: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    deleted_chunks: int
    source: str


# --------------------------------------------------------------------- 系统
class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    llm_model: str
    embedding_model: str
    indexed_chunks: int
    hybrid_search: bool
    rerank: bool
