"""
agent/audio_agent.py – Agentic audio analysis orchestrator.

The AudioAnalysisAgent drives the entire analysis pipeline using Gemini
function calling.  The LLM decides which tools to call, in what order,
and when to stop – producing a rich, structured report at the end.

Architecture
------------
  AudioAnalysisAgent
GeminiClient         (LLM with function-calling)
    ├── MetadataExtractor    (ffprobe → metadata dict)
    ├── QualityAnalyzer      (ffmpeg filters → quality dicts)
    ├── RulesEngine          (deterministic issue detection)
    └── ReportGenerator      (JSON + LLM insight assembly)

The agent loop:
  1. Sends an initial analysis request to Gemini.
  2. Receives function_call parts from the model.
  3. Dispatches each call to the appropriate tool method.
  4. Returns function_response parts to the model.
  5. Repeats until the model produces no more function calls.
  6. Extracts and returns the final report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.genai import types as genai_types

from agent.tool_definitions import TOOL_DECLARATIONS
from analyzers.metadata_extractor import MetadataExtractor
from analyzers.quality_analyzer import QualityAnalyzer
from analyzers.rules_engine import RulesEngine
from llm.gemini_client import GeminiClient
from llm.prompts import AGENT_SYSTEM_PROMPT
from reports.report_generator import ReportGenerator
from utils.logger import get_logger

logger = get_logger(__name__)

# Maximum number of agentic iterations (safety circuit-breaker)
_MAX_ITERATIONS = 20


class AudioAnalysisAgent:
    """
    Agentic orchestrator that uses Gemini + function calling to analyse
    audio files and produce comprehensive quality reports.

    Example
    -------
    agent = AudioAnalysisAgent()
    report = agent.analyze("/recordings/deposition_001.wav")
    print(json.dumps(report, indent=2))
    """

    def __init__(self) -> None:
        self._llm = GeminiClient()
        self._metadata = MetadataExtractor()
        self._quality = QualityAnalyzer()
        self._rules = RulesEngine()
        self._reporter = ReportGenerator(llm_client=self._llm)

        # Build the Gemini model with tool declarations
        self._model = self._llm.build_agent_model(
            tool_declarations=TOOL_DECLARATIONS,
            system_instruction=AGENT_SYSTEM_PROMPT,
        )

        # Map tool names → Python methods
        self._dispatch: dict[str, Any] = {
            "extract_audio_metadata": self._t_extract_metadata,
            "detect_silence_segments": self._t_detect_silence,
            "analyze_volume_levels": self._t_analyze_volume,
            "detect_audio_clipping": self._t_detect_clipping,
            "analyze_audio_statistics": self._t_analyze_stats,
            "run_rules_based_analysis": self._t_rules_analysis,
            "generate_final_report": self._t_generate_report,
        }

    # Public API
    def analyze(self, file_path: str | Path) -> dict[str, Any]:
        """
        Run the full agentic analysis pipeline on *file_path*.

        Parameters
        ----------
        file_path : Path to any FFmpeg-supported audio file.

        Returns
        -------
        Structured analysis report (dict, JSON-serialisable).
        """
        path = str(file_path)
        logger.info("═" * 60)
        logger.info("AudioAnalysisAgent starting analysis: %s", path)
        logger.info("═" * 60)

        # Shared state accumulator – populated by tool calls
        collected: dict[str, Any] = {}

        # Start a multi-turn chat session
        chat = self._model.start_chat()

        initial_message = (
            f"Please perform a full audio quality analysis on the file:\n"
            f"  {path}\n\n"
            "Follow the standard workflow:\n"
            "  1. extract_audio_metadata\n"
            "  2. detect_silence_segments\n"
            "  3. analyze_volume_levels\n"
            "  4. detect_audio_clipping\n"
            "  5. analyze_audio_statistics\n"
            "  6. run_rules_based_analysis\n"
            "  7. generate_final_report\n\n"
            "Be systematic and do not skip any step."
        )

        response = chat.send_message(initial_message)

        # Agentic loop
        for iteration in range(1, _MAX_ITERATIONS + 1):
            fn_calls = self._llm.extract_function_calls(response)

            if not fn_calls:
                logger.info(
                    "Agent completed in %d iteration(s) – no more function calls.",
                    iteration,
                )
                break

            logger.info(
                "Iteration %d/%d – model issued %d function call(s).",
                iteration, _MAX_ITERATIONS, len(fn_calls),
            )

            # Execute every function call the model issued
            response_parts: list[protos.Part] = []
            for fc in fn_calls:
                tool_result = self._invoke_tool(fc, collected)
                response_parts.append(
                    GeminiClient.make_function_response(fc.name, tool_result)
                )

            # Return results to the model
            response = chat.send_message(response_parts)

        else:
            logger.warning(
                "Agent hit the maximum iteration limit (%d). "
                "Generating report from collected data.",
                _MAX_ITERATIONS,
            )

        # Extract final report
        if "final_report" in collected:
            logger.info("Final report extracted from agent state.")
            return collected["final_report"]

        # Fallback: build the report directly from whatever was collected
        logger.warning("Final report not found in state – building from scratch.")
        return self._reporter.build(path, collected)

    # Tool dispatcher
    def _invoke_tool(
        self,
        fc: genai_types.FunctionCall,
        state: dict[str, Any],
    ) -> Any:
        """Dispatch a function call to the matching tool method."""
        name = fc.name
        args = dict(fc.args) if fc.args else {}

        logger.info("  ▶ Tool: %-35s args: %s", name, list(args.keys()))

        handler = self._dispatch.get(name)
        if handler is None:
            msg = f"Unknown tool requested by model: '{name}'"
            logger.error(msg)
            return {"error": msg}

        try:
            result = handler(state=state, **args)
            return result
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception: %s", name, exc)
            return {"error": str(exc)}

    # Tool implementations
    def _t_extract_metadata(self, state: dict, file_path: str, **_) -> dict:
        data = self._metadata.extract(file_path)
        state["metadata"] = data
        state["file_path"] = file_path
        return data

    def _t_detect_silence(
        self,
        state: dict,
        file_path: str,
        threshold_db: float | None = None,
        min_duration: float | None = None,
        **_,
    ) -> dict:
        kwargs: dict[str, Any] = {}
        if threshold_db is not None:
            kwargs["threshold_db"] = float(threshold_db)
        if min_duration is not None:
            kwargs["min_duration"] = float(min_duration)
        data = self._quality.detect_silence(file_path, **kwargs)
        state["silence_analysis"] = data
        return data

    def _t_analyze_volume(self, state: dict, file_path: str, **_) -> dict:
        data = self._quality.analyze_volume(file_path)
        state["volume_analysis"] = data
        return data

    def _t_detect_clipping(self, state: dict, file_path: str, **_) -> dict:
        data = self._quality.detect_clipping(file_path)
        state["clipping_analysis"] = data
        return data

    def _t_analyze_stats(self, state: dict, file_path: str, **_) -> dict:
        data = self._quality.analyze_stats(file_path)
        state["audio_statistics"] = data
        return data

    def _t_rules_analysis(
        self, state: dict, analysis_data_json: str, **_
    ) -> dict:
        try:
            extra = json.loads(analysis_data_json)
            # Merge agent-provided data with accumulated state
            merged = {**state, **extra}
        except (json.JSONDecodeError, TypeError):
            merged = state

        data = self._rules.analyze(merged)
        state["rules_analysis"] = data
        return data

    def _t_generate_report(
        self, state: dict, file_path: str, all_analysis_data_json: str, **_
    ) -> dict:
        try:
            extra = json.loads(all_analysis_data_json)
            merged = {**state, **extra}
        except (json.JSONDecodeError, TypeError):
            merged = state

        report = self._reporter.build(file_path, merged)
        state["final_report"] = report
        return report
