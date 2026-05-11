import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

def seconds_to_hms(seconds: float) -> str:
    """Convert a duration in seconds to HH:MM:SS string."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_bytes(num_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes //= 1024
    return f"{num_bytes:.1f} TB"


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.utcnow().isoformat() + "Z"


def file_checksum(path: str | Path, algorithm: str = "sha256") -> str:
    """Compute a hex checksum for a file (for deduplication)."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_json_loads(text: str) -> Any:
    """Parse JSON, stripping markdown code fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    return json.loads(cleaned.strip())


def sanitize_filename(name: str) -> str:
    """Replace characters that are unsafe in filenames."""
    for char in r'\/:*?"<>|':
        name = name.replace(char, "_")
    return name
