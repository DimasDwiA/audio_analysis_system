"""
api/routes/analysis.py – Single-file and live-recording analysis endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from agent.audio_agent import AudioAnalysisAgent
from api.schemas.models import (
    AnalyzeFileRequest,
    AnalysisReportResponse,
    RecordAndAnalyzeRequest,
)
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/analyze", tags=["Analysis"])

# Shared agent instance (avoids re-initialising Gemini on every request)
_agent = AudioAnalysisAgent()

@router.post(
    "/file",
    summary="Analyse an audio file by server-side path",
    response_model=dict,
)
async def analyze_file(body: AnalyzeFileRequest):
    """
    Trigger a full agentic analysis on a file that already exists on the server.

    Returns the complete analysis report as JSON.
    """
    path = Path(body.file_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {body.file_path}",
        )

    logger.info("API /analyze/file → %s", path)
    try:
        report = _agent.analyze(path)
    except Exception as exc:
        logger.exception("Analysis failed for %s", path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    if body.save_report:
        out = settings.OUTPUT_DIR / f"{path.stem}_report.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Report saved: %s", out)

    return report

@router.post(
    "/upload",
    summary="Upload an audio file and analyse it",
    response_model=dict,
)
async def analyze_upload(file: UploadFile = File(...)):
    """
    Accept a multipart file upload, save it temporarily, run analysis,
    and return the report.
    """
    # Save the uploaded file
    upload_path = settings.TEMP_DIR / file.filename
    content = await file.read()
    upload_path.write_bytes(content)
    logger.info("Uploaded file saved: %s (%d bytes)", upload_path, len(content))

    try:
        report = _agent.analyze(upload_path)
    except Exception as exc:
        logger.exception("Analysis failed for upload: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    finally:
        # Clean up temp file
        try:
            upload_path.unlink()
        except OSError:
            pass

    return report

@router.post(
    "/record",
    summary="Record audio from microphone then analyse it",
    response_model=dict,
)
async def record_and_analyze(body: RecordAndAnalyzeRequest):
    """
    Record *duration_seconds* of audio from the server's default microphone,
    then run the full analysis pipeline on it.

    > ⚠️  Requires `sounddevice` and a connected microphone on the server.
    """
    try:
        from recorder.live_recorder import LiveRecorder
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="sounddevice / soundfile not installed.  "
                   "Run: pip install sounddevice soundfile",
        )

    recorder = LiveRecorder(device=body.device)
    try:
        wav_path = recorder.record_fixed(body.duration_seconds, body.filename)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recording failed: {exc}",
        )

    try:
        report = _agent.analyze(wav_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {exc}",
        )

    return report