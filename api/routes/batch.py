"""
api/routes/batch.py – Batch analysis endpoints.
"""

from __future__ import annotations
from fastapi import APIRouter, HTTPException, status

from api.schemas.models import BatchAnalyzeRequest
from processor.batch_processor import BatchProcessor
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/batch", tags=["Batch"])

_processor = BatchProcessor()

@router.post(
    "/analyze",
    summary="Analyse multiple audio files in one request",
    response_model=dict,
)
async def batch_analyze(body: BatchAnalyzeRequest):
    """
    Run the full analysis pipeline on every file in *file_paths* and
    return a consolidated batch report.

    All files must be accessible on the server's file system.
    """
    logger.info("API /batch/analyze → %d files", len(body.file_paths))

    try:
        report = _processor.process_files(
            file_paths=body.file_paths,
            generate_batch_summary=body.generate_batch_summary,
        )
    except Exception as exc:
        logger.exception("Batch processing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return report

@router.get(
    "/devices",
    summary="List available audio input devices",
    response_model=list,
)
async def list_devices():
    """Return a list of audio input devices visible to the server."""
    try:
        from recorder.live_recorder import LiveRecorder
        return LiveRecorder.list_devices()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )