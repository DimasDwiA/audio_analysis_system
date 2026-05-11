"""
analyzers/ffmpeg_runner.py – Low-level FFmpeg / FFprobe command executor.

This module is the single entry-point for all subprocess calls to ffmpeg
and ffprobe.  Every other analyzer imports *this* module rather than
calling subprocess directly, making it easy to swap the backend or add
retries / timeouts in one place.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class FFmpegError(RuntimeError):
    """Raised when an FFmpeg/FFprobe command exits with a non-zero status."""


class FFmpegRunner:
    """
    Thin wrapper around ffmpeg and ffprobe that executes commands and
    returns their output as strings or parsed structures.
    """

    def __init__(self) -> None:
        self.ffmpeg = settings.FFMPEG_PATH
        self.ffprobe = settings.FFPROBE_PATH
        self._verify_installation()

    
    # Public helpers
    def probe_json(self, file_path: str | Path) -> dict[str, Any]:
        """
        Run ``ffprobe`` on *file_path* and return the parsed JSON payload
        containing stream and format information.
        """
        cmd = [
            self.ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(file_path),
        ]
        logger.debug("ffprobe command: %s", " ".join(cmd))
        raw = self._run(cmd)
        return json.loads(raw)

    def run_filter(
        self,
        file_path: str | Path,
        audio_filter: str,
        extra_args: list[str] | None = None,
    ) -> str:
        """
        Apply *audio_filter* to *file_path* with ``ffmpeg -af`` and capture
        stderr (where ffmpeg writes filter statistics).

        Returns the combined stderr output as a string.
        """
        cmd = [
            self.ffmpeg,
            "-i", str(file_path),
            "-af", audio_filter,
            "-f", "null",
            "-",
        ]
        if extra_args:
            cmd.extend(extra_args)

        logger.debug("ffmpeg filter command: %s", " ".join(cmd))
        return self._run_stderr(cmd)

    def run_custom(self, args: list[str]) -> tuple[str, str]:
        """
        Execute an arbitrary ffmpeg/ffprobe command described by *args*
        (the full argument list including the binary name).

        Returns ``(stdout, stderr)``.
        """
        logger.debug("Custom FFmpeg command: %s", " ".join(args))
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout, result.stderr

    # Internal helpers
    def _run(self, cmd: list[str]) -> str:
        """Run a command and return stdout; raise FFmpegError on failure."""
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise FFmpegError(
                f"Command failed (exit {result.returncode}): "
                f"{' '.join(cmd)}\nstderr: {result.stderr[:2000]}"
            )
        return result.stdout

    def _run_stderr(self, cmd: list[str]) -> str:
        """
        Run a command and return *stderr* (many ffmpeg filters write their
        output to stderr even on success).
        """
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # ffmpeg returns 0 or 1 for filter-only runs; accept both
        if result.returncode not in (0, 1):
            raise FFmpegError(
                f"Command failed (exit {result.returncode}): "
                f"{' '.join(cmd)}\nstderr: {result.stderr[:2000]}"
            )
        return result.stderr

    def _verify_installation(self) -> None:
        """Check that ffmpeg and ffprobe are available; log their versions."""
        for binary in (self.ffmpeg, self.ffprobe):
            try:
                result = subprocess.run(
                    [binary, "-version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                first_line = result.stdout.split("\n")[0]
                logger.debug("%s found: %s", binary, first_line)
            except FileNotFoundError:
                raise FFmpegError(
                    f"'{binary}' not found. Install FFmpeg and ensure it is "
                    "in your PATH (or set FFMPEG_PATH / FFPROBE_PATH in .env)."
                )
