"""
processor/batch_processor.py – Batch audio analysis.

Processes multiple audio files (or live-recorded chunks) sequentially,
aggregates results, and optionally generates a cross-file batch summary
using the LLM.

Features
--------
  * Processes files in order with per-file progress reporting.
  * Stores every individual report in memory and optionally to disk.
  * Calls Gemini to generate a batch-level summary after all files finish.
  * Handles individual-file failures gracefully (skips and records error).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent.audio_agent import AudioAnalysisAgent
from llm.gemini_client import GeminiClient
from llm.prompts import BATCH_SUMMARY_PROMPT, INSIGHT_SYSTEM_PROMPT
from utils.helpers import sanitize_filename, utc_now_iso
from utils.logger import get_logger

logger = get_logger(__name__)


class BatchProcessor:
    """
    Analyse multiple audio files and produce a consolidated batch report.

    Parameters
    ----------
    output_dir : Directory where per-file and batch reports are written.
    save_json  : If True, write each individual report as a JSON file.
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        save_json: bool = True,
    ) -> None:
        from config import settings  # local to avoid circular import
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_json = save_json
        self._agent = AudioAnalysisAgent()
        self._llm = GeminiClient()

    # Public API
    def process_files(
        self,
        file_paths: list[str | Path],
        generate_batch_summary: bool = True,
    ) -> dict[str, Any]:
        """
        Analyse each file in *file_paths* and return the combined batch report.

        Parameters
        ----------
        file_paths             : List of paths to audio files.
        generate_batch_summary : Call the LLM for a cross-file summary.

        Returns
        -------
        Batch report dict with keys:
            batch_id, generated_at, total_files, processed, failed,
            individual_reports, batch_summary (optional)
        """
        batch_id = f"batch_{int(time.time())}"
        logger.info("━" * 60)
        logger.info("Batch '%s': %d file(s) queued.", batch_id, len(file_paths))
        logger.info("━" * 60)

        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for idx, fp in enumerate(file_paths, start=1):
            path = Path(fp)
            logger.info("[%d/%d] Processing: %s", idx, len(file_paths), path.name)

            try:
                report = self._agent.analyze(path)

                if self.save_json:
                    self._save_report(report, path.stem)

                results.append(report)
                logger.info(
                    "  ✓ Done – score=%s, grade=%s, issues=%d",
                    report.get("audio_quality", {}).get("quality_score", "?"),
                    report.get("audio_quality", {}).get("quality_grade", "?"),
                    len(report.get("issues", [])),
                )

            except Exception as exc:
                logger.error("  ✗ Failed: %s – %s", path.name, exc)
                errors.append({"file": str(path), "error": str(exc)})

        # Batch summary
        batch_summary: dict[str, Any] = {}
        if generate_batch_summary and results:
            batch_summary = self._generate_batch_summary(results)

        # Aggregate statistics
        scores = [
            r.get("audio_quality", {}).get("quality_score", 0)
            for r in results
        ]

        batch_report: dict[str, Any] = {
            "batch_id": batch_id,
            "generated_at": utc_now_iso(),
            "total_files": len(file_paths),
            "processed": len(results),
            "failed": len(errors),
            "avg_quality_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "min_quality_score": min(scores) if scores else 0,
            "max_quality_score": max(scores) if scores else 0,
            "errors": errors,
            "individual_reports": results,
            "batch_summary": batch_summary,
        }

        if self.save_json:
            self._save_report(batch_report, batch_id)

        logger.info(
            "Batch complete – %d/%d succeeded, avg_score=%.1f",
            len(results), len(file_paths),
            batch_report["avg_quality_score"],
        )
        return batch_report
    
    # Private helpers
    def _save_report(self, report: dict[str, Any], stem: str) -> Path:
        """Write *report* to a JSON file in output_dir."""
        safe_stem = sanitize_filename(stem)
        out_path = self.output_dir / f"{safe_stem}_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.debug("Report saved: %s", out_path)
        return out_path

    def _generate_batch_summary(
        self, reports: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Call Gemini to summarise the batch results."""
        logger.info("Generating batch summary via LLM…")

        summaries = []
        for r in reports:
            aq = r.get("audio_quality", {})
            summaries.append(
                f"- {r.get('file_name', 'unknown')}: "
                f"score={aq.get('quality_score','?')}, "
                f"grade={aq.get('quality_grade','?')}, "
                f"issues={len(r.get('issues', []))}, "
                f"clipping={aq.get('clipping_detected', False)}, "
                f"silence_ratio={aq.get('silence_ratio', 0):.2%}"
            )

        prompt = BATCH_SUMMARY_PROMPT.format(
            file_count=len(reports),
            individual_summaries="\n".join(summaries),
        )

        try:
            raw = self._llm.complete(prompt, system_instruction=INSIGHT_SYSTEM_PROMPT)
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            return json.loads(cleaned.strip())
        except Exception as exc:
            logger.error("Batch LLM summary failed: %s", exc)
            return {"error": str(exc)}
