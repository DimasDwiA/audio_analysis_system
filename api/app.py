"""
api/app.py – FastAPI application factory.

Creates and configures the FastAPI instance, registers all routers,
and adds middleware.  Import ``app`` from here for ASGI deployment.
"""

from __future__ import annotations

import subprocess
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.analysis import router as analysis_router
from api.routes.batch import router as batch_router
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Lifespan (startup / shutdown)
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup / shutdown logic."""
    logger.info("Audio Analysis API starting…")
    logger.info("Output directory: %s", settings.OUTPUT_DIR)
    logger.info("Gemini model: %s", settings.GEMINI_MODEL)
    yield
    logger.info("Audio Analysis API shutting down.")

# App factory
def create_app() -> FastAPI:
    app = FastAPI(
        title="Audio Analysis API",
        description=(
            "Agentic audio quality analysis powered by FFmpeg + Gemini LLM. "
            "Supports single-file, batch, and live-recording analysis."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS – allow all origins for team collaboration (restrict in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(analysis_router)
    app.include_router(batch_router)

    # Health check
    @app.get("/health", tags=["System"])
    async def health():
        """System health check – verifies FFmpeg and Gemini availability."""
        ffmpeg_ok = _check_binary(settings.FFMPEG_PATH)
        return JSONResponse({
            "status": "ok",
            "version": "1.0.0",
            "ffmpeg_available": ffmpeg_ok,
            "gemini_configured": bool(settings.GEMINI_API_KEY),
            "output_dir": str(settings.OUTPUT_DIR),
        })

    @app.get("/", tags=["System"])
    async def root():
        return {"message": "Audio Analysis API", "docs": "/docs"}

    return app

def _check_binary(binary: str) -> bool:
    try:
        subprocess.run(
            [binary, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        return True
    except Exception:
        return False

# Module-level app instance for `uvicorn api.app:app`
app = create_app()
