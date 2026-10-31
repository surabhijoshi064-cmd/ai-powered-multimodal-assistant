import os
from pathlib import Path
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Secure Multimodal AI Chatbot"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "127.0.0.1"

    # Database
    DATABASE_URL: str = "sqlite:///./storage/local_chatbot.db"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    # CORS
    CORS_ORIGINS: Union[str, List[str]] = "http://localhost:8501,http://127.0.0.1:8501"

    @property
    def cors_origins_list(self) -> List[str]:
        if isinstance(self.CORS_ORIGINS, list):
            return self.CORS_ORIGINS
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    # File uploads & storage
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_DOCUMENT_EXTENSIONS: str = "pdf,txt,py,js,ts,html,css,json,csv,md,sql"
    ALLOWED_IMAGE_EXTENSIONS: str = "png,jpg,jpeg"
    ALLOWED_AUDIO_EXTENSIONS: str = "wav,mp3,m4a,ogg"
    
    STORAGE_DIR: str = "storage"
    UPLOAD_DIR: str = "storage/uploads"
    AUDIO_DIR: str = "storage/audio"

    @property
    def allowed_doc_exts(self) -> List[str]:
        return [ext.strip().lower().lstrip(".") for ext in self.ALLOWED_DOCUMENT_EXTENSIONS.split(",")]

    @property
    def allowed_img_exts(self) -> List[str]:
        return [ext.strip().lower().lstrip(".") for ext in self.ALLOWED_IMAGE_EXTENSIONS.split(",")]

    @property
    def allowed_aud_exts(self) -> List[str]:
        return [ext.strip().lower().lstrip(".") for ext in self.ALLOWED_AUDIO_EXTENSIONS.split(",")]

    # AI Providers
    LLM_PROVIDER: str = "mock"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""

    # Gemini (Google) — set GEMINI_API_KEY to enable; overrides LLM_PROVIDER automatically
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_RPM_LIMIT: int = 4       # requests per minute (free tier = 5; use 4 for safety margin)
    GEMINI_MAX_RETRIES: int = 3     # max retry attempts on 429 ResourceExhausted

    # Ollama (local AI) — fallback when Gemini quota is exhausted or unavailable
    # Set OLLAMA_ENABLED=true and ensure `ollama serve` is running.
    OLLAMA_ENABLED: bool = False
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma2:2b"          # default model (small, fast, fits ~4 GB RAM)
    OLLAMA_TIMEOUT: float = 120.0            # seconds; large models may take longer on first call
    OLLAMA_FALLBACK_ON_DAILY_QUOTA: bool = True   # auto-switch to Ollama on Gemini daily 429

    @field_validator("GEMINI_MODEL", mode="before")
    @classmethod
    def normalize_gemini_model(cls, v: str) -> str:
        if v and v.startswith("models/"):
            return v[len("models/"):]
        return v or "gemini-1.5-flash"

    VISION_PROVIDER: str = "mock"
    VISION_MODEL: str = "gpt-4o-mini"
    VISION_API_KEY: str = ""

    TRANSCRIPTION_PROVIDER: str = "mock"
    TRANSCRIPTION_MODEL: str = "whisper-1"
    TRANSCRIPTION_API_KEY: str = ""

    TTS_PROVIDER: str = "mock"
    TTS_MODEL: str = "tts-1"
    TTS_VOICE: str = "alloy"
    TTS_API_KEY: str = ""

    # Embeddings & RAG
    EMBEDDING_PROVIDER: str = "mock"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_API_KEY: str = ""

    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 100
    RAG_TOP_K: int = 4
    RAG_SIMILARITY_THRESHOLD: float = 0.35

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        """Ensure necessary storage directories exist."""
        for d in [self.STORAGE_DIR, self.UPLOAD_DIR, self.AUDIO_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)


# Instantiate singleton settings
settings = Settings()
settings.ensure_directories()
