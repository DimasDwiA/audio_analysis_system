"""
analyzers/rules_engine.py – Rule-based audio quality issue detection.

This module applies a deterministic set of rules to the raw metrics
collected by the other analyzers.  It produces:

  * A list of structured ``Issue`` objects (each with severity + message)
  * A quality score  (0–100)
  * A quality grade  (A / B / C / D / F)

The output is kept separate from the LLM so the final report can combine
rule precision with LLM-generated prose.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# Data types
@dataclass
class Issue:
    """A single detected audio quality issue."""
    severity: str          # "critical" | "warning" | "info"
    category: str          # "silence" | "clipping" | "volume" | "noise" | "format"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class RulesReport:
    """Aggregated output from the rules engine."""
    issues: list[Issue]
    quality_score: int          # 0–100
    quality_grade: str          # A / B / C / D / F
    summary_flags: list[str]    # short machine-readable flags

    def to_dict(self) -> dict[str, Any]:
        return {
            "issues": [i.to_dict() for i in self.issues],
            "quality_score": self.quality_score,
            "quality_grade": self.quality_grade,
            "summary_flags": self.summary_flags,
        }

# Engine
class RulesEngine:
    """
    Deterministic rule evaluator.

    Accepts the raw collected data dict (as produced by the audio agent)
    and returns a :class:`RulesReport`.
    """

    def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Run all rules against *data* and return the serialised report.

        Parameters
        ----------
        data : Combined analysis data containing keys such as
               ``metadata``, ``silence_analysis``, ``volume_analysis``,
               ``clipping_analysis``, ``audio_statistics``.
        """
        issues: list[Issue] = []
        flags: list[str] = []

        metadata = data.get("metadata", {})
        silence = data.get("silence_analysis", {})
        volume = data.get("volume_analysis", {})
        clipping = data.get("clipping_analysis", {})
        stats = data.get("audio_statistics", {})

        duration = metadata.get("duration_seconds", 0)

        # ── Rule set ──────────────────────────────────────────────────────────
        issues += self._check_silence(silence, duration, flags)
        issues += self._check_volume(volume, flags)
        issues += self._check_clipping(clipping, flags)
        issues += self._check_noise(stats, flags)
        issues += self._check_format(metadata, flags)

        score = self._compute_score(issues)
        grade = self._score_to_grade(score)

        logger.info(
            "Rules analysis complete – %d issues, score=%d, grade=%s",
            len(issues), score, grade,
        )

        report = RulesReport(
            issues=issues,
            quality_score=score,
            quality_grade=grade,
            summary_flags=flags,
        )
        return report.to_dict()

    # Individual rule groups
    def _check_silence(
        self, silence: dict, duration: float, flags: list[str]
    ) -> list[Issue]:
        issues: list[Issue] = []
        if not silence:
            return issues

        segments = silence.get("segments", [])
        total_silence = silence.get("total_silence_seconds", 0)
        silence_ratio = total_silence / duration if duration > 0 else 0

        # Flag extended individual silence segments (>30 s)
        for seg in segments:
            seg_dur = seg.get("duration_s") or 0
            if seg_dur >= 30:
                issues.append(Issue(
                    severity="critical",
                    category="silence",
                    message=(
                        f"Extended silence detected: {seg_dur:.1f}s "
                        f"between {seg['start_s']:.1f}s – {seg.get('end_s', '?')}s"
                    ),
                    detail=seg,
                ))
            elif seg_dur >= 10:
                issues.append(Issue(
                    severity="warning",
                    category="silence",
                    message=(
                        f"Long silence detected: {seg_dur:.1f}s "
                        f"starting at {seg['start_s']:.1f}s"
                    ),
                    detail=seg,
                ))

        # Flag high overall silence ratio
        if silence_ratio > settings.HIGH_SILENCE_RATIO_THRESHOLD:
            pct = silence_ratio * 100
            flags.append("HIGH_SILENCE_RATIO")
            issues.append(Issue(
                severity="warning",
                category="silence",
                message=(
                    f"High silence ratio: {pct:.1f}% of total audio is silent "
                    f"({total_silence:.1f}s / {duration:.1f}s)"
                ),
                detail={
                    "silence_ratio": round(silence_ratio, 4),
                    "total_silence_s": total_silence,
                    "duration_s": duration,
                },
            ))

        if silence.get("segment_count", 0) > 10:
            flags.append("MANY_SILENCE_SEGMENTS")
            issues.append(Issue(
                severity="info",
                category="silence",
                message=(
                    f"Fragmented audio: {silence['segment_count']} silence "
                    "segments detected – consider noise gating or trimming."
                ),
                detail={"segment_count": silence["segment_count"]},
            ))

        return issues

    def _check_volume(self, volume: dict, flags: list[str]) -> list[Issue]:
        issues: list[Issue] = []
        if not volume:
            return issues

        mean_db = volume.get("mean_volume_db")
        max_db = volume.get("max_volume_db")

        if volume.get("is_too_quiet") and mean_db is not None:
            flags.append("TOO_QUIET")
            issues.append(Issue(
                severity="warning",
                category="volume",
                message=(
                    f"Audio is too quiet: mean volume {mean_db:.1f} dB "
                    f"(threshold {settings.LOW_VOLUME_THRESHOLD_DB} dB). "
                    "Consider applying normalisation or gain."
                ),
                detail={"mean_volume_db": mean_db},
            ))

        # Headroom check: max volume very low means possible gain issues
        if max_db is not None and max_db < -20:
            flags.append("LOW_HEADROOM")
            issues.append(Issue(
                severity="info",
                category="volume",
                message=(
                    f"Peak volume is only {max_db:.1f} dB – "
                    "significant headroom unused.  Normalise to -3 dB FS recommended."
                ),
                detail={"max_volume_db": max_db},
            ))

        return issues

    def _check_clipping(self, clipping: dict, flags: list[str]) -> list[Issue]:
        issues: list[Issue] = []
        if not clipping:
            return issues

        if clipping.get("clipping_detected"):
            flags.append("CLIPPING_DETECTED")
            peak = clipping.get("peak_level_db", 0)
            issues.append(Issue(
                severity="critical",
                category="clipping",
                message=(
                    f"Audio clipping detected! Peak level: {peak:.1f} dBFS. "
                    "Clipping causes irreversible distortion.  "
                    "Reduce recording gain or use a limiter."
                ),
                detail={
                    "peak_level_db": peak,
                    "peak_count": clipping.get("peak_count"),
                },
            ))

        return issues

    def _check_noise(self, stats: dict, flags: list[str]) -> list[Issue]:
        issues: list[Issue] = []
        if not stats:
            return issues

        noise_floor = stats.get("noise_floor_db")
        dynamic_range = stats.get("dynamic_range_db")
        dc_offset = stats.get("dc_offset")

        if stats.get("is_noisy") and noise_floor is not None:
            flags.append("HIGH_NOISE_FLOOR")
            issues.append(Issue(
                severity="warning",
                category="noise",
                message=(
                    f"Elevated noise floor: {noise_floor:.1f} dB "
                    f"(threshold {settings.NOISE_FLOOR_THRESHOLD_DB} dB). "
                    "Consider applying a noise reduction filter."
                ),
                detail={"noise_floor_db": noise_floor},
            ))

        if dynamic_range is not None and dynamic_range < 10:
            flags.append("LOW_DYNAMIC_RANGE")
            issues.append(Issue(
                severity="info",
                category="noise",
                message=(
                    f"Low dynamic range: {dynamic_range:.1f} dB – "
                    "audio may sound compressed or over-processed."
                ),
                detail={"dynamic_range_db": dynamic_range},
            ))

        if dc_offset is not None and abs(dc_offset) > 0.01:
            flags.append("DC_OFFSET")
            issues.append(Issue(
                severity="warning",
                category="noise",
                message=(
                    f"DC offset detected: {dc_offset:.4f}. "
                    "This can cause clicks on edit boundaries.  "
                    "Apply a high-pass filter at 20 Hz to remove it."
                ),
                detail={"dc_offset": dc_offset},
            ))

        return issues

    def _check_format(self, metadata: dict, flags: list[str]) -> list[Issue]:
        issues: list[Issue] = []
        if not metadata:
            return issues

        sr = metadata.get("sample_rate_hz", 0)
        channels = metadata.get("channels", 0)
        bitrate = metadata.get("overall_bitrate_kbps", 0)
        duration = metadata.get("duration_seconds", 0)

        # Sample rate
        if sr > 0 and sr < 22050:
            flags.append("LOW_SAMPLE_RATE")
            issues.append(Issue(
                severity="warning",
                category="format",
                message=(
                    f"Low sample rate: {sr} Hz.  "
                    "Speech is acceptable at 16 kHz but 44.1 kHz is recommended for music."
                ),
                detail={"sample_rate_hz": sr},
            ))

        # Channels
        if channels == 0:
            issues.append(Issue(
                severity="critical",
                category="format",
                message="No audio channels detected – file may be corrupt.",
                detail={},
            ))

        # Bitrate (lossy formats)
        codec = metadata.get("codec_name", "")
        if codec in ("mp3", "aac", "ogg") and bitrate > 0 and bitrate < 64:
            flags.append("LOW_BITRATE")
            issues.append(Issue(
                severity="warning",
                category="format",
                message=(
                    f"Low bitrate for lossy codec ({codec}): {bitrate:.0f} kbps. "
                    "Audio quality may be degraded.  128 kbps minimum recommended."
                ),
                detail={"codec": codec, "bitrate_kbps": bitrate},
            ))

        # Very short file
        if 0 < duration < 1:
            issues.append(Issue(
                severity="info",
                category="format",
                message=f"Very short audio file: {duration:.2f}s.",
                detail={"duration_s": duration},
            ))

        return issues

    # Scoring helpers
    @staticmethod
    def _compute_score(issues: list[Issue]) -> int:
        """Compute a 0–100 quality score from the issue list."""
        penalties = {"critical": 25, "warning": 10, "info": 2}
        total_penalty = sum(penalties.get(i.severity, 0) for i in issues)
        return max(0, 100 - total_penalty)

    @staticmethod
    def _score_to_grade(score: int) -> str:
        if score >= 90:
            return "A"
        if score >= 75:
            return "B"
        if score >= 60:
            return "C"
        if score >= 40:
            return "D"
        return "F"