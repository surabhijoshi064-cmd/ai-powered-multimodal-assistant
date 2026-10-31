"""
Ollama Local AI Service for MindfulAI.

Responsibilities:
  - Check whether Ollama CLI is installed on Windows.
  - Check whether the Ollama server is running at OLLAMA_BASE_URL.
  - Discover locally-pulled models.
  - Provide text generation and vision understanding (when supported by model).
  - Explicitly verify model capabilities: Never claim Gemma 2B supports images or audio.
  - Provide actionable instructions when the server is stopped or a model is missing.
"""
import shutil
import logging
import base64
from typing import Dict, List, Optional, Tuple, Any

import httpx

import config
from services.quota_manager import quota_manager
from app.utils.ollama_client import (
    check_ollama_health,
    list_ollama_models,
    chat_with_ollama,
    chat_with_ollama_native,
    model_supports_vision,
    OllamaUnavailableError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaResponseError,
)

logger = logging.getLogger(__name__)


class OllamaServiceError(RuntimeError):
    """Base exception for Ollama service."""


class OllamaNotInstalledError(OllamaServiceError):
    """Raised when Ollama executable is not found in system PATH."""


class OllamaServerOfflineError(OllamaServiceError):
    """Raised when Ollama server is not running on localhost."""


class OllamaModelMissingError(OllamaServiceError):
    """Raised when the configured model has not been downloaded yet."""


class OllamaUnsupportedModalityError(OllamaServiceError):
    """Raised when attempting multimodal input on a text-only model."""


class OllamaService:
    """Service wrapping local Ollama AI interactions with accurate capability checking."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        text_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.text_model = text_model or config.OLLAMA_TEXT_MODEL
        self.vision_model = vision_model or config.OLLAMA_VISION_MODEL
        self.timeout = timeout or config.OLLAMA_TIMEOUT

    # ── Health & Availability Checks ─────────────────────────────────────

    def is_installed(self) -> bool:
        """Check if `ollama` CLI executable is installed on the system."""
        return shutil.which("ollama") is not None or shutil.which("ollama.exe") is not None

    def is_running(self, timeout: float = 2.0) -> bool:
        """Check if Ollama server is active and responding to HTTP requests."""
        return check_ollama_health(self.base_url, timeout=timeout)

    def list_models(self, timeout: float = 4.0) -> List[str]:
        """Return names of all locally pulled Ollama models."""
        return list_ollama_models(self.base_url, timeout=timeout)

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model tag is already pulled locally."""
        pulled = self.list_models()
        return any(
            m == model_name or m.startswith(f"{model_name}:") or model_name.startswith(f"{m}:")
            for m in pulled
        )

    def get_status_info(self) -> Dict[str, Any]:
        """Return comprehensive status dict for UI indicators."""
        installed = self.is_installed()
        running = self.is_running()
        models = self.list_models() if running else []
        text_ready = any(self.text_model in m for m in models) if running else False

        return {
            "installed": installed,
            "running": running,
            "base_url": self.base_url,
            "configured_model": self.text_model,
            "models_available": models,
            "model_ready": text_ready,
            "supports_vision": model_supports_vision(self.text_model),
        }

    # ── Text Generation ──────────────────────────────────────────────────

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> Tuple[str, str]:
        """
        Generate text response from multi-turn chat messages using local Ollama.
        Returns: (response_text, model_used)
        """
        if not self.is_running():
            raise OllamaServerOfflineError(
                f"Ollama server is not running at {self.base_url}.\n"
                "To start Ollama, run: `ollama serve` in a Windows PowerShell terminal."
            )

        target_model = model_name or self.text_model

        try:
            text, used_model = chat_with_ollama(
                messages=messages,
                model=target_model,
                system_prompt=system_instruction,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            quota_manager.set_active_provider("ollama")
            return text, used_model

        except OllamaModelNotFoundError as e:
            raise OllamaModelMissingError(
                f"Ollama model '{target_model}' is not downloaded.\n"
                f"Run in PowerShell: `ollama pull {target_model}`"
            ) from e

        except OllamaUnavailableError as e:
            raise OllamaServerOfflineError(str(e)) from e

        except OllamaTimeoutError as e:
            raise OllamaServiceError(f"Ollama request timed out after {self.timeout}s: {e}") from e

        except Exception as e:
            # Fall back to native /api/chat endpoint
            try:
                text, used_model = chat_with_ollama_native(
                    messages=messages,
                    model=target_model,
                    system_prompt=system_instruction,
                    temperature=temperature,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
                quota_manager.set_active_provider("ollama")
                return text, used_model
            except Exception as native_err:
                raise OllamaServiceError(f"Ollama error: {native_err}") from native_err

    # ── Image Understanding (Vision Fallback) ─────────────────────────────

    def analyze_image(
        self,
        image_bytes: bytes,
        user_prompt: str = "Describe this image in detail.",
        model_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Analyze an image with a local vision model (e.g. llava, moondream).
        Raises OllamaUnsupportedModalityError if model does not support vision.
        """
        target_model = model_name or self.vision_model

        if not model_supports_vision(target_model):
            raise OllamaUnsupportedModalityError(
                f"The configured model '{target_model}' is a text-only model and does not support image analysis.\n"
                "To enable local vision analysis, install a multimodal model:\n"
                "  `ollama pull moondream` (lightweight, ~1.8 GB)\n"
                "  or `ollama pull llava:7b` (~4.7 GB)"
            )

        if not self.is_running():
            raise OllamaServerOfflineError(f"Ollama server is not running at {self.base_url}.")

        # Send image to Ollama's native /api/generate or /api/chat with base64 image
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": target_model,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [b64_img],
                }
            ],
            "stream": False,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, json=payload)
                res.raise_for_status()
                data = res.json()
                text = data.get("message", {}).get("content", "").strip()
                quota_manager.set_active_provider("ollama")
                return text, target_model
        except Exception as e:
            raise OllamaServiceError(f"Local vision analysis failed with model '{target_model}': {e}") from e

