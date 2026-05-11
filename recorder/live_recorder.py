"""
recorder/live_recorder.py – Real-time audio capture from a microphone.

Records audio from the default (or specified) input device and saves it
as a WAV file that can then be passed to the analysis pipeline.

Supports two recording modes:
  1. Fixed duration  – records for N seconds then stops.
  2. Chunked stream  – records indefinitely in chunks; returns chunk paths
     so the caller can analyse them in near-real-time.
"""

from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Callable, Generator

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class LiveRecorder:
    """
    Records audio from the system microphone and saves it to WAV files.

    Parameters
    ----------
    sample_rate     : Recording sample rate in Hz.
    channels        : Number of input channels (1 = mono, 2 = stereo).
    output_dir      : Directory where WAV files are saved.
    chunk_seconds   : Duration of each chunk when streaming.
    device          : sounddevice device index/name (None = system default).
    """

    def __init__(
        self,
        sample_rate: int | None = None,
        channels: int | None = None,
        output_dir: Path | None = None,
        chunk_seconds: int | None = None,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate or settings.SAMPLE_RATE
        self.channels = channels or settings.CHANNELS
        self.output_dir = output_dir or settings.TEMP_DIR
        self.chunk_seconds = chunk_seconds or settings.RECORD_CHUNK_SECONDS
        self.device = device
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._stop_event = threading.Event()
    
    
    # Public methods
    def record_fixed(self, duration_seconds: int, filename: str | None = None) -> Path:
        """
        Record *duration_seconds* of audio and return the path to the WAV file.

        Parameters
        ----------
        duration_seconds : How many seconds to record.
        filename         : Optional output filename (auto-generated if None).
        """
        if filename is None:
            filename = f"recording_{int(time.time())}.wav"

        out_path = self.output_dir / filename
        logger.info(
            "Recording %.0fs at %d Hz, %d ch → %s",
            duration_seconds, self.sample_rate, self.channels, out_path,
        )

        audio = sd.rec(
            frames=int(duration_seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
        )
        sd.wait()  

        sf.write(str(out_path), audio, self.sample_rate, subtype="PCM_16")
        logger.info("Recording saved: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
        return out_path

    def record_stream(
        self,
        on_chunk: Callable[[Path], None] | None = None,
        max_chunks: int | None = None,
    ) -> Generator[Path, None, None]:
        """
        Record continuously in chunks of *chunk_seconds* each.

        Yields the path to each completed chunk WAV file.
        If *on_chunk* is provided it is called with the path immediately
        (useful for running analysis in a background thread).

        Call :meth:`stop` from another thread to end the stream.

        Parameters
        ----------
        on_chunk   : Callback invoked with each chunk path as it completes.
        max_chunks : Stop after this many chunks (None = run until stop()).
        """
        self._stop_event.clear()
        chunk_idx = 0

        logger.info(
            "Starting live stream recording (chunk=%ds, sr=%d, ch=%d)",
            self.chunk_seconds, self.sample_rate, self.channels,
        )

        while not self._stop_event.is_set():
            if max_chunks is not None and chunk_idx >= max_chunks:
                logger.info("Reached max_chunks=%d – stopping stream.", max_chunks)
                break

            chunk_path = self.output_dir / f"live_chunk_{int(time.time())}_{chunk_idx}.wav"
            logger.info("Recording chunk %d → %s", chunk_idx, chunk_path.name)

            audio = sd.rec(
                frames=int(self.chunk_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
            )
            sd.wait()

            if self._stop_event.is_set():
                break  

            sf.write(str(chunk_path), audio, self.sample_rate, subtype="PCM_16")
            logger.info("Chunk %d saved: %s", chunk_idx, chunk_path.name)

            if on_chunk:
                on_chunk(chunk_path)

            yield chunk_path
            chunk_idx += 1

        logger.info("Live stream recording ended after %d chunks.", chunk_idx)

    def stop(self) -> None:
        """Signal the streaming recorder to stop after the current chunk."""
        logger.info("Stop signal sent to live recorder.")
        self._stop_event.set()

    # Utility
    @staticmethod
    def list_devices() -> list[dict]:
        """Return a list of available audio input devices."""
        devices = sd.query_devices()
        result = []
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                result.append({
                    "index": idx,
                    "name": dev["name"],
                    "max_input_channels": dev["max_input_channels"],
                    "default_samplerate": dev["default_samplerate"],
                })
        return result
