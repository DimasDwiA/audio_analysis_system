"""
analyzers/quality_analyzer.py – FFmpeg-based audio quality analysis.

Provides four independent analyses:
  1. Silence detection  (silencedetect filter)
  2. Volume levels      (volumedetect filter)
  3. Clipping detection (astats filter – peak sample values)
  4. Audio statistics   (astats filter – full per-channel stats)

Each method returns a plain dict so it can be serialised to JSON directly.
"""

import re
from pathlib import Path
from typing import Any

from analyzers.ffmpeg_runner import FFmpegRunner
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class QualityAnalyzer:
    """Run FFmpeg quality-analysis filters and return structured results."""

    def __init__(self) -> None:
        self._runner = FFmpegRunner()
    
    # 1. Silence detection
    def detect_silence(
        self,
        file_path: str | Path,
        threshold_db: float | None = None,
        min_duration: float | None = None,
    ) -> dict[str, Any]:
        """
        Detect silence segments using the ``silencedetect`` filter.

        Parameters
        ----------
        file_path     : Path to the audio file.
        threshold_db  : dB level below which audio is considered silent.
                        Defaults to settings.SILENCE_THRESHOLD_DB.
        min_duration  : Minimum silence length in seconds.
                        Defaults to settings.SILENCE_MIN_DURATION.

        Returns a dict with:
            threshold_db, min_duration_s, segments (list of dicts),
            total_silence_seconds, silence_ratio, segment_count
        """
        thr = threshold_db if threshold_db is not None else settings.SILENCE_THRESHOLD_DB
        dur = min_duration if min_duration is not None else settings.SILENCE_MIN_DURATION

        logger.info(
            "Detecting silence in '%s' (threshold=%.1f dB, min_dur=%.1fs)",
            Path(file_path).name, thr, dur,
        )

        af_filter = f"silencedetect=noise={thr}dB:d={dur}"
        stderr = self._runner.run_filter(file_path, af_filter)

        segments = self._parse_silence(stderr)

        # Compute total duration for silence ratio
        # (we'll refine the ratio in the report generator once we have duration)
        total_silence = sum(s["duration_s"] for s in segments)

        return {
            "threshold_db": thr,
            "min_duration_s": dur,
            "segments": segments,
            "total_silence_seconds": round(total_silence, 3),
            "segment_count": len(segments),
        }

    @staticmethod
    def _parse_silence(stderr: str) -> list[dict[str, Any]]:
        """Parse silence_start / silence_end / silence_duration from stderr."""
        starts: list[float] = [
            float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", stderr)
        ]
        ends: list[float] = [
            float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", stderr)
        ]
        durations: list[float] = [
            float(m) for m in re.findall(r"silence_duration:\s*([\d.]+)", stderr)
        ]

        segments: list[dict[str, Any]] = []
        for i, start in enumerate(starts):
            end = ends[i] if i < len(ends) else None
            dur = durations[i] if i < len(durations) else (end - start if end else None)
            segments.append({
                "start_s": round(start, 3),
                "end_s": round(end, 3) if end is not None else None,
                "duration_s": round(dur, 3) if dur is not None else None,
            })
        return segments

    # 2. Volume analysis
    def analyze_volume(self, file_path: str | Path) -> dict[str, Any]:
        """
        Measure volume statistics using the ``volumedetect`` filter.

        Returns a dict with:
            mean_volume_db, max_volume_db, histogram_db (list),
            is_too_quiet (bool)
        """
        logger.info("Analyzing volume levels in '%s'", Path(file_path).name)

        stderr = self._runner.run_filter(file_path, "volumedetect")

        mean_db = self._parse_float(r"mean_volume:\s*([-\d.]+)\s*dB", stderr)
        max_db = self._parse_float(r"max_volume:\s*([-\d.]+)\s*dB", stderr)

        # Histogram bins
        histogram: list[dict[str, Any]] = []
        for m in re.finditer(r"histogram_(\d+)db:\s*(\d+)", stderr):
            histogram.append({"db": -int(m.group(1)), "count": int(m.group(2))})

        is_too_quiet = (
            mean_db is not None and mean_db < settings.LOW_VOLUME_THRESHOLD_DB
        )

        return {
            "mean_volume_db": mean_db,
            "max_volume_db": max_db,
            "histogram_db": histogram,
            "is_too_quiet": is_too_quiet,
            "low_volume_threshold_db": settings.LOW_VOLUME_THRESHOLD_DB,
        }

    # 3. Clipping detection
    def detect_clipping(self, file_path: str | Path) -> dict[str, Any]:
        """
        Detect potential clipping / distortion using the ``astats`` filter.

        A peak value ≥ CLIPPING_PEAK_FRACTION of the maximum possible
        amplitude indicates clipping.  We also look at the number of
        samples that hit the absolute peak.

        Returns a dict with:
            clipping_detected (bool), peak_level_db, max_difference,
            peak_count, clipping_fraction, details
        """
        logger.info("Detecting clipping in '%s'", Path(file_path).name)

        # astats writes per-channel statistics
        stderr = self._runner.run_filter(
            file_path,
            "astats=metadata=1:reset=1",
        )

        peak_db = self._parse_float(r"Peak level dB:\s*([-\d.]+)", stderr)
        rms_db = self._parse_float(r"RMS level dB:\s*([-\d.]+)", stderr)
        peak_count = self._parse_int(r"Peak count:\s*(\d+)", stderr)
        max_diff = self._parse_float(r"Max difference:\s*([\d.]+)", stderr)

        # Clipping heuristic: Peak level dB is 0 dBFS (full scale)
        clipping_detected = peak_db is not None and peak_db >= -0.1

        return {
            "clipping_detected": clipping_detected,
            "peak_level_db": peak_db,
            "rms_level_db": rms_db,
            "peak_count": peak_count,
            "max_difference": max_diff,
            "clipping_threshold_db": -0.1,
        }

    # 4. Full audio statistics
    def analyze_stats(self, file_path: str | Path) -> dict[str, Any]:
        """
        Collect detailed audio statistics with ``astats``.

        Returns a dict with:
            dc_offset, rms_db, crest_factor_db, flat_factor,
            noise_floor_db, dynamic_range_db, estimated_noise_profile
        """
        logger.info("Collecting audio statistics for '%s'", Path(file_path).name)

        stderr = self._runner.run_filter(file_path, "astats")

        rms_db = self._parse_float(r"RMS level dB:\s*([-\d.]+)", stderr)
        rms_peak_db = self._parse_float(r"RMS peak dB:\s*([-\d.]+)", stderr)
        rms_trough_db = self._parse_float(r"RMS trough dB:\s*([-\d.]+)", stderr)
        crest_db = self._parse_float(r"Crest factor:\s*([-\d.]+)", stderr)
        flat_factor = self._parse_float(r"Flat factor:\s*([-\d.]+)", stderr)
        dc_offset = self._parse_float(r"DC offset:\s*([-\d.e+]+)", stderr)
        noise_floor_db = self._parse_float(r"Noise floor dB:\s*([-\d.]+)", stderr)
        dynamic_range_db = self._parse_float(r"Dynamic range:\s*([-\d.]+)", stderr)

        is_noisy = (
            noise_floor_db is not None
            and noise_floor_db > settings.NOISE_FLOOR_THRESHOLD_DB
        )

        return {
            "rms_level_db": rms_db,
            "rms_peak_db": rms_peak_db,
            "rms_trough_db": rms_trough_db,
            "crest_factor_db": crest_db,
            "flat_factor": flat_factor,
            "dc_offset": dc_offset,
            "noise_floor_db": noise_floor_db,
            "dynamic_range_db": dynamic_range_db,
            "is_noisy": is_noisy,
            "noise_floor_threshold_db": settings.NOISE_FLOOR_THRESHOLD_DB,
        }

    # Private parsing helpers
    @staticmethod
    def _parse_float(pattern: str, text: str) -> float | None:
        m = re.search(pattern, text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    @staticmethod
    def _parse_int(pattern: str, text: str) -> int | None:
        m = re.search(pattern, text)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
        return None
