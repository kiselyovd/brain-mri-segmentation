"""Pydantic request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    version: str


class SegmentationResponse(BaseModel):
    mask_base64: str = Field(..., description="PNG mask, base64-encoded")
