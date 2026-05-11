"""
reports/report_generator.py – Final report assembly.

Combines raw metrics (from FFmpeg) with rule-based issues and
LLM-generated human-readable insights into one canonical JSON report.

Output schema (matches the project spec):
{
  "report_id": "uuid",
  "generated_at": "ISO-8601",
  "file_name": "deposition_001.wav",
  "file_path": "/absolute/path",
  "duration_seconds": 3600,
  "audio_quality": {
    "silence_ratio": 0.12,
    "clipping_detected": false,
    "avg_volume_db": -18,
    "peak_volume_db": -6,
    "noise_floor_db": -60,
    "dynamic_range_db": 54,
    "sample_rate_hz": 44100,
    "channels": 1,
    "codec": "pcm_s16le",
    "bitrate_kbps": 705.6,
    "quality_score": 85,
    "quality_grade": "B"
  },
  "silence_segments": [...],
  "issues": ["Long silence detected between 1200–1500s"],
  "llm_insights": {
    "executive_summary": "...",
    "recommended_actions": [...],
    "overall_verdict": "..."
  }
}
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm.prompts import INSIGHT_SYSTEM_PROMPT, INSIGHT_USER_PROMPT
from utils.helpers import utc_now_iso
from utils.logger import get_logger

if TYPE_CHECKING:
    from llm.gemini_client import GeminiClient

logger = get_logger(__name__)

class ReportGenerator:
    """Assembles the final structured report from raw analysis data."""

    def __init__(self, llm_client: GeminiClient | None = None) -> None:
        self._llm = llm_client

    def build(
        self,
        file_path: str | Path,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build the final report dict.

        Parameters
        ----------
        file_path : Path to the audio file (used for metadata fallback).
        data      : Aggregated state dict from the agent containing keys:
                    metadata, silence_analysis, volume_analysis,
                    clipping_analysis, audio_statistics, rules_analysis.
        """
        path = Path(file_path)

        metadata = data.get("metadata", {})
        silence = data.get("silence_analysis", {})
        volume = data.get("volume_analysis", {})
        clipping = data.get("clipping_analysis", {})
        stats = data.get("audio_statistics", {})
        rules = data.get("rules_analysis", {})

        duration = metadata.get("duration_seconds", 0)
        total_silence = silence.get("total_silence_seconds", 0)
        silence_ratio = round(total_silence / duration, 4) if duration > 0 else 0.0

        # Audio quality block
        audio_quality = {
            "silence_ratio": silence_ratio,
            "total_silence_seconds": total_silence,
            "silence_segment_count": silence.get("segment_count", 0),
            "clipping_detected": clipping.get("clipping_detected", False),
            "avg_volume_db": volume.get("mean_volume_db"),
            "peak_volume_db": volume.get("max_volume_db"),
            "noise_floor_db": stats.get("noise_floor_db"),
            "dynamic_range_db": stats.get("dynamic_range_db"),
            "rms_level_db": stats.get("rms_level_db"),
            "dc_offset": stats.get("dc_offset"),
            "sample_rate_hz": metadata.get("sample_rate_hz"),
            "channels": metadata.get("channels"),
            "channel_layout": metadata.get("channel_layout"),
            "codec": metadata.get("codec_name"),
            "bitrate_kbps": metadata.get("overall_bitrate_kbps"),
            "quality_score": rules.get("quality_score", 0),
            "quality_grade": rules.get("quality_grade", "?"),
        }

        # Issue messages (human-readable strings)
        issues_raw = rules.get("issues", [])
        issue_messages = [i["message"] for i in issues_raw if "message" in i]

        # LLM insights
        llm_insights: dict[str, Any] = {}
        if self._llm:
            llm_insights = self._generate_llm_insights(
                file_path=str(path),
                audio_quality=audio_quality,
                issues=issues_raw,
                quality_score=rules.get("quality_score", 0),
                quality_grade=rules.get("quality_grade", "?"),
            )

        # Assemble final report
        report: dict[str, Any] = {
            "report_id": str(uuid.uuid4()),
            "generated_at": utc_now_iso(),
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "duration_seconds": duration,
            "duration_hms": metadata.get("duration_hms", ""),
            "file_size_human": metadata.get("file_size_human", ""),
            "format": metadata.get("format_name", ""),
            "audio_quality": audio_quality,
            "silence_segments": silence.get("segments", []),
            "issues": issue_messages,
            "rules_analysis": {
                "quality_score": rules.get("quality_score", 0),
                "quality_grade": rules.get("quality_grade", "?"),
                "summary_flags": rules.get("summary_flags", []),
                "detailed_issues": issues_raw,
            },
            "raw_metrics": {
                "metadata": metadata,
                "silence_analysis": silence,
                "volume_analysis": volume,
                "clipping_analysis": clipping,
                "audio_statistics": stats,
            },
            "llm_insights": llm_insights,
        }

        logger.info(
            "Report assembled for '%s' – score=%s, grade=%s, issues=%d",
            path.name,
            rules.get("quality_score", "?"),
            rules.get("quality_grade", "?"),
            len(issue_messages),
        )
        return report

    # LLM insight generation
    def _generate_llm_insights(
        self,
        file_path: str,
        audio_quality: dict,
        issues: list[dict],
        quality_score: int,
        quality_grade: str,
    ) -> dict[str, Any]:
        """Call Gemini to generate human-readable insights."""
        logger.info("Generating LLM insights via Gemini…")

        issues_text = "\n".join(
            f"  [{i.get('severity','?').upper()}] ({i.get('category','?')}) "
            f"{i.get('message','')}"
            for i in issues
        ) or "  No issues detected."

        prompt = INSIGHT_USER_PROMPT.format(
            report_json=json.dumps({"file": file_path, "audio_quality": audio_quality}, indent=2),
            issues_text=issues_text,
            quality_score=quality_score,
            quality_grade=quality_grade,
        )

        try:
            raw = self._llm.complete(prompt, system_instruction=INSIGHT_SYSTEM_PROMPT)
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            return json.loads(cleaned.strip())
        except Exception as exc:
            logger.error("LLM insight generation failed: %s", exc)
            return {
                "executive_summary": "LLM insight generation unavailable.",
                "recommended_actions": [],
                "overall_verdict": "Unable to generate LLM verdict.",
                "error": str(exc),
            }
