"""
analyzers/metadata_extractor.py – Audio metadata extraction via ffprobe.

Extracts duration, bitrate, sample rate, channels, codec, format, and
file-level information from any audio format that FFmpeg supports.
"""

from pathlib import Path
from typing import Any

from analyzers.ffmpeg_runner import FFmpegRunner
from utils.helpers import format_bytes, seconds_to_hms
from utils.logger import get_logger

logger = get_logger(__name__)

class MetadataExtractor:
    """Extract structured metadata from an audio file using ffprobe."""

    def __init__(self) -> None:
        self._runner = FFmpegRunner()

    def extract(self, file_path: str | Path) -> dict[str, Any]:
        """
        Run ffprobe on *file_path* and return a flat metadata dictionary.

        Returns
        -------
        dict with keys:
            file_name, file_path, file_size_bytes, file_size_human,
            format_name, duration_seconds, duration_hms,
            overall_bitrate_bps, overall_bitrate_kbps,
            codec_name, codec_long_name,
            sample_rate_hz, channels, channel_layout,
            bit_depth, stream_bitrate_bps
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        logger.info("Extracting metadata from: %s", path.name)

        raw = self._runner.probe_json(path)
        fmt = raw.get("format", {})
        streams = raw.get("streams", [])

        # Pick the first audio stream
        audio_stream = next(
            (s for s in streams if s.get("codec_type") == "audio"), {}
        )

        duration = float(fmt.get("duration") or audio_stream.get("duration") or 0)
        bitrate = int(fmt.get("bit_rate") or audio_stream.get("bit_rate") or 0)
        file_size = int(fmt.get("size") or path.stat().st_size)

        metadata = {
            # File info
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "file_size_bytes": file_size,
            "file_size_human": format_bytes(file_size),
            # Container / format
            "format_name": fmt.get("format_long_name", fmt.get("format_name", "unknown")),
            "format_tags": fmt.get("tags", {}),
            # Duration
            "duration_seconds": round(duration, 3),
            "duration_hms": seconds_to_hms(duration),
            # Bitrate
            "overall_bitrate_bps": bitrate,
            "overall_bitrate_kbps": round(bitrate / 1000, 1) if bitrate else 0,
            # Stream-level audio info
            "codec_name": audio_stream.get("codec_name", "unknown"),
            "codec_long_name": audio_stream.get("codec_long_name", "unknown"),
            "sample_rate_hz": int(audio_stream.get("sample_rate") or 0),
            "channels": int(audio_stream.get("channels") or 0),
            "channel_layout": audio_stream.get("channel_layout", "unknown"),
            "bit_depth": audio_stream.get("bits_per_sample")
                         or audio_stream.get("bits_per_raw_sample"),
            "stream_bitrate_bps": int(audio_stream.get("bit_rate") or 0),
        }

        logger.info(
            "Metadata OK – duration=%.1fs, codec=%s, sr=%dHz, ch=%d",
            metadata["duration_seconds"],
            metadata["codec_name"],
            metadata["sample_rate_hz"],
            metadata["channels"],
        )
        return metadata
