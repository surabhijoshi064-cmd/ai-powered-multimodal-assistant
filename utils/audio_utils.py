"""
Audio utility functions for format conversion and validation.
"""
import io
import wave
from typing import Optional, Tuple


def get_audio_info(audio_bytes: bytes) -> dict:
    """Extract sample rate, channels, and duration from WAV audio bytes."""
    try:
        with io.BytesIO(audio_bytes) as buf:
            with wave.open(buf, "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                duration = nframes / float(framerate)
                return {
                    "channels": channels,
                    "sample_width": sample_width,
                    "framerate": framerate,
                    "duration_seconds": round(duration, 2),
                    "valid_wav": True,
                }
    except Exception:
        return {"valid_wav": False, "size_bytes": len(audio_bytes)}
