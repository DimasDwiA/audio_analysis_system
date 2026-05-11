"""
config.py – Central application configuration.

All settings are read from environment variables (or .env file).
Import the singleton `settings` object anywhere in the codebase:

    from config import settings
    print(settings.GEMINI_API_KEY)
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini LLM
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # FFmpeg
    FFMPEG_PATH: str = "ffmpeg"
    FFPROBE_PATH: str = "ffprobe"

    # API Server 
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_SECRET_KEY: str = "change_me_in_production"

    # Audio Recording 
    SAMPLE_RATE: int = 44100
    CHANNELS: int = 1
    RECORD_CHUNK_SECONDS: int = 30

    # Analysis Thresholds 
    SILENCE_THRESHOLD_DB: float = -30.0
    SILENCE_MIN_DURATION: float = 2.0
    LOW_VOLUME_THRESHOLD_DB: float = -40.0
    CLIPPING_PEAK_FRACTION: float = 0.99
    HIGH_SILENCE_RATIO_THRESHOLD: float = 0.20
    NOISE_FLOOR_THRESHOLD_DB: float = -50.0

    # Storage 
    OUTPUT_DIR: Path = Path("./output")
    TEMP_DIR: Path = Path("./temp")

    def ensure_directories(self) -> None:
        """Create output and temp directories if they don't exist."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Singleton – import this everywhere
settings = Settings()
settings.ensure_directories()
