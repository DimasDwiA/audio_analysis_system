"""
api/schemas/models.py – Pydantic request / response models for the REST API.
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field

# Request models
class AnalyzeFileRequest(BaseModel):
    """Body for POST /analyze/file"""
    file_path: str = Field(..., description="Absolute path to an audio file on the server.")
    save_report: bool = Field(True, description="Save JSON report to output directory.")

class AnalyzeUrlRequest(BaseModel):
    """Body for POST /analyze/url (future extension)"""
    url: str
    save_report: bool = True

class BatchAnalyzeRequest(BaseModel):
    """Body for POST /batch/analyze"""
    file_paths: list[str] = Field(..., min_length=1)
    generate_batch_summary: bool = True
    save_reports: bool = True

class RecordAndAnalyzeRequest(BaseModel):
    """Body for POST /analyze/record"""
    duration_seconds: int = Field(30, ge=1, le=3600)
    filename: str | None = None
    device: int | None = None

# Response models
class AudioQualityResponse(BaseModel):
    silence_ratio: float | None
    clipping_detected: bool | None
    avg_volume_db: float | None
    peak_volume_db: float | None
    noise_floor_db: float | None
    dynamic_range_db: float | None
    quality_score: int | None
    quality_grade: str | None
    sample_rate_hz: int | None
    channels: int | None
    codec: str | None
    bitrate_kbps: float | None

class LLMInsightsResponse(BaseModel):
    executive_summary: str = ""
    recommended_actions: list[str] = []
    overall_verdict: str = ""

class AnalysisReportResponse(BaseModel):
    report_id: str
    generated_at: str
    file_name: str
    file_path: str
    duration_seconds: float
    duration_hms: str
    audio_quality: AudioQualityResponse
    issues: list[str]
    llm_insights: LLMInsightsResponse
    # Full raw data available too
    raw: dict[str, Any] = Field(default_factory=dict)

class BatchReportResponse(BaseModel):
    batch_id: str
    generated_at: str
    total_files: int
    processed: int
    failed: int
    avg_quality_score: float
    individual_reports: list[dict[str, Any]]
    batch_summary: dict[str, Any]
    errors: list[dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    version: str
    ffmpeg_available: bool
    gemini_configured: bool

class ErrorResponse(BaseModel):
    error: str
    detail: str = ""