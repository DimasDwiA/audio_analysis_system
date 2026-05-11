"""
agent/tool_definitions.py – Gemini FunctionDeclaration definitions.

Uses the new google-genai SDK (google.genai.types).
"""

from google.genai import types

# Shorthand
_Schema = types.Schema
_T = types.Type


def _str_param(description: str) -> _Schema:
    return _Schema(type=_T.STRING, description=description)


def _num_param(description: str) -> _Schema:
    return _Schema(type=_T.NUMBER, description=description)


TOOL_DECLARATIONS: list[types.FunctionDeclaration] = [

    types.FunctionDeclaration(
        name="extract_audio_metadata",
        description=(
            "Extract basic audio metadata from a file using ffprobe. "
            "Returns duration, bitrate, sample rate, channels, codec, and file info."
        ),
        parameters=_Schema(
            type=_T.OBJECT,
            properties={
                "file_path": _str_param("Absolute or relative path to the audio file."),
            },
            required=["file_path"],
        ),
    ),

    types.FunctionDeclaration(
        name="detect_silence_segments",
        description=(
            "Detect silence segments in the audio using FFmpeg silencedetect filter. "
            "Returns start/end timestamps, durations, and summary statistics."
        ),
        parameters=_Schema(
            type=_T.OBJECT,
            properties={
                "file_path": _str_param("Path to the audio file."),
                "threshold_db": _num_param(
                    "dB level below which audio is considered silent (default: -30)."
                ),
                "min_duration": _num_param(
                    "Minimum silence segment length in seconds (default: 2.0)."
                ),
            },
            required=["file_path"],
        ),
    ),

    types.FunctionDeclaration(
        name="analyze_volume_levels",
        description=(
            "Measure audio volume levels using FFmpeg volumedetect filter. "
            "Returns mean and peak volume in dB and a histogram."
        ),
        parameters=_Schema(
            type=_T.OBJECT,
            properties={
                "file_path": _str_param("Path to the audio file."),
            },
            required=["file_path"],
        ),
    ),

    types.FunctionDeclaration(
        name="detect_audio_clipping",
        description=(
            "Detect audio clipping or distortion using FFmpeg astats filter. "
            "Returns peak level in dBFS and whether clipping is present."
        ),
        parameters=_Schema(
            type=_T.OBJECT,
            properties={
                "file_path": _str_param("Path to the audio file."),
            },
            required=["file_path"],
        ),
    ),

    types.FunctionDeclaration(
        name="analyze_audio_statistics",
        description=(
            "Collect detailed audio statistics (RMS, crest factor, noise floor, "
            "dynamic range, DC offset) using FFmpeg astats filter."
        ),
        parameters=_Schema(
            type=_T.OBJECT,
            properties={
                "file_path": _str_param("Path to the audio file."),
            },
            required=["file_path"],
        ),
    ),

    types.FunctionDeclaration(
        name="run_rules_based_analysis",
        description=(
            "Apply a deterministic rules engine to the collected metrics. "
            "Returns structured issues with severity levels and a quality score."
        ),
        parameters=_Schema(
            type=_T.OBJECT,
            properties={
                "analysis_data_json": _str_param(
                    "JSON string of the combined analysis data collected so far."
                ),
            },
            required=["analysis_data_json"],
        ),
    ),

    types.FunctionDeclaration(
        name="generate_final_report",
        description=(
            "Compile all analysis results into the final structured JSON report, "
            "including LLM-generated insights and recommended actions."
        ),
        parameters=_Schema(
            type=_T.OBJECT,
            properties={
                "file_path": _str_param("Path to the original audio file."),
                "all_analysis_data_json": _str_param(
                    "JSON string of every analysis result collected so far."
                ),
            },
            required=["file_path", "all_analysis_data_json"],
        ),
    ),
]
