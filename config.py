"""
Centralized Configuration for MindfulAI Multimodal Chatbot.

Defines model identities, provider configurations, and file limits.
All values are configurable via the root .env file.
"""
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv()

# ---------------------------------------------------------------------------
# Application Settings
# ---------------------------------------------------------------------------
APP_NAME: str = os.getenv("APP_NAME", "MindfulAI — Multimodal AI Chatbot")
APP_ENV: str = os.getenv("APP_ENV", "development")
DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")

# Storage Directories
STORAGE_DIR: str = os.getenv("STORAGE_DIR", "storage")
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "storage/uploads")
AUDIO_DIR: str = os.getenv("AUDIO_DIR", "storage/audio")

for d in [STORAGE_DIR, UPLOAD_DIR, AUDIO_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# File Validation Limits
MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB
ALLOWED_DOCUMENT_EXTENSIONS: List[str] = [
    ext.strip().lower().lstrip(".")
    for ext in os.getenv("ALLOWED_DOCUMENT_EXTENSIONS", "pdf,txt,py,js,ts,html,css,json,csv,md,sql").split(",")
]
ALLOWED_IMAGE_EXTENSIONS: List[str] = [
    ext.strip().lower().lstrip(".")
    for ext in os.getenv("ALLOWED_IMAGE_EXTENSIONS", "png,jpg,jpeg,webp").split(",")
]
ALLOWED_AUDIO_EXTENSIONS: List[str] = [
    ext.strip().lower().lstrip(".")
    for ext in os.getenv("ALLOWED_AUDIO_EXTENSIONS", "wav,mp3,m4a,ogg,flac,webm").split(",")
]

# ---------------------------------------------------------------------------
# Google Gemini API Configuration (Primary Provider)
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()

# Clean models helper
def _clean_model(name: str, fallback: str) -> str:
    val = (name or "").strip()
    if val.startswith("models/"):
        val = val[len("models/"):]
    return val or fallback

# Centralized Model Identifiers
# CRITICAL: Separate models by modality!
# Never send audio input to TTS_MODEL!
TEXT_MODEL: str = _clean_model(
    os.getenv("GEMINI_TEXT_MODEL", os.getenv("GEMINI_MODEL", "gemini-1.5-flash")),
    "gemini-1.5-flash",
)
IMAGE_MODEL: str = _clean_model(
    os.getenv("GEMINI_IMAGE_MODEL", "gemini-1.5-flash"),
    "gemini-1.5-flash",
)
AUDIO_INPUT_MODEL: str = _clean_model(
    os.getenv("GEMINI_AUDIO_INPUT_MODEL", "gemini-1.5-flash"),
    "gemini-1.5-flash",
)
TTS_MODEL: str = _clean_model(
    os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts"),
    "gemini-2.5-flash-preview-tts",
)
IMAGEN_MODEL: str = _clean_model(
    os.getenv("GEMINI_IMAGEN_MODEL", "imagen-3.0-generate-002"),
    "imagen-3.0-generate-002",
)
GEMINI_PRO_IMAGE_MODEL: str = _clean_model(
    os.getenv("GEMINI_PRO_IMAGE_MODEL", "gemini-3-pro-image"),
    "gemini-3-pro-image",
)

# Safety check: if user configured a -tts model as TEXT_MODEL or AUDIO_INPUT_MODEL, auto-correct
if "-tts" in TEXT_MODEL.lower():
    TEXT_MODEL = "gemini-1.5-flash"
if "-tts" in AUDIO_INPUT_MODEL.lower():
    AUDIO_INPUT_MODEL = "gemini-1.5-flash"

GEMINI_RPM_LIMIT: int = int(os.getenv("GEMINI_RPM_LIMIT", "4"))
GEMINI_MAX_RETRIES: int = int(os.getenv("GEMINI_MAX_RETRIES", "3"))

# ---------------------------------------------------------------------------
# Ollama Configuration (Local AI — Secondary / Automatic Fallback)
# ---------------------------------------------------------------------------
OLLAMA_ENABLED: bool = os.getenv("OLLAMA_ENABLED", "true").lower() in ("true", "1", "yes")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_TEXT_MODEL: str = os.getenv("OLLAMA_TEXT_MODEL", os.getenv("OLLAMA_MODEL", "gemma2:2b"))
OLLAMA_VISION_MODEL: str = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")
OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))
OLLAMA_FALLBACK_ON_DAILY_QUOTA: bool = os.getenv("OLLAMA_FALLBACK_ON_DAILY_QUOTA", "true").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# ComfyUI Configuration (Local Image Generation & Editing Engine)
# ---------------------------------------------------------------------------
COMFYUI_ENABLED: bool = os.getenv("COMFYUI_ENABLED", "false").lower() in ("true", "1", "yes")
COMFYUI_BASE_URL: str = os.getenv("COMFYUI_BASE_URL", "http://127.0.0.1:8188").rstrip("/")
COMFYUI_TIMEOUT: float = float(os.getenv("COMFYUI_TIMEOUT", "180.0"))
COMFYUI_MODEL: str = os.getenv("COMFYUI_MODEL", "v1-5-pruned-emaonly.safetensors")

# ---------------------------------------------------------------------------
# Quota Management Settings
# ---------------------------------------------------------------------------
# Cooldown duration in seconds when Google API does not provide an exact Retry-After duration.
# Google resets free-tier daily quotas at midnight Pacific; default cooldown is 3600s (1 hr)
QUOTA_DEFAULT_COOLDOWN_SECONDS: float = float(os.getenv("QUOTA_DEFAULT_COOLDOWN_SECONDS", "3600.0"))

