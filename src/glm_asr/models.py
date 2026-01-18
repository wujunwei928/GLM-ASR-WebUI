"""Pydantic 数据模型"""

from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    """转录响应模型"""

    success: bool
    text: str
    duration: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """健康检查响应模型"""

    status: str
    model_loaded: bool
    device: str
