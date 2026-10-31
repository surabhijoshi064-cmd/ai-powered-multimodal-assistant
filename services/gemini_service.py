"""
Google Gemini AI Service.

Handles text generation, image analysis, audio input processing, and image generation.
Enforces strict model-modality separation:
  - Audio input is ONLY routed to models verified to accept audio (e.g. gemini-1.5-flash)
  - NEVER routes incoming audio to a Text-to-Speech (-tts) model.
  - Integrates with quota_manager to track daily quota and rate limits.
"""
import io
import time
import random
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

import config
from services.quota_manager import (
    quota_manager,
    is_daily_quota_error,
    is_rate_limit_error,
    parse_retry_after,
)

logger = logging.getLogger(__name__)


class GeminiServiceError(RuntimeError):
    """Base exception for Gemini service errors."""


class GeminiDailyQuotaExhaustedError(GeminiServiceError):
    """Raised when Gemini daily free-tier quota is reached."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class GeminiRateLimitExhaustedError(GeminiServiceError):
    """Raised when short-term RPM limit is reached and retries exhausted."""
    def __init__(self, message: str, retry_after: float = 60.0):
        super().__init__(message)
        self.retry_after = retry_after


class GeminiAuthFailureError(GeminiServiceError):
    """Raised on HTTP 401 or invalid API key."""


class GeminiModalityMismatchError(GeminiServiceError):
    """Raised when an unsupported modality is passed to a model (e.g. audio to TTS model)."""


class GeminiService:
    """Encapsulates all Google Gemini API interactions with modality validation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or config.GEMINI_API_KEY).strip()
        self._genai = None

    def _get_genai(self):
        if self._genai is None:
            try:
                import google.generativeai as genai
                if not self.api_key:
                    raise GeminiAuthFailureError("GEMINI_API_KEY is not set.")
                genai.configure(api_key=self.api_key)
                self._genai = genai
            except ImportError:
                raise GeminiServiceError("google-generativeai is not installed.")
        return self._genai

    # ── Text Generation ──────────────────────────────────────────────────

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Tuple[str, str]:
        """
        Generate text response from multi-turn chat messages.
        Returns: (response_text, model_used)
        """
        target_model = model_name or config.TEXT_MODEL
        # Guard: NEVER use a -tts model for general text chat
        if "-tts" in target_model.lower():
            target_model = config.TEXT_MODEL

        # Format history for Gemini chat
        history = []
        for msg in messages[:-1]:
            role = "user" if msg.get("role") == "user" else "model"
            content = msg.get("content", "")
            if content:
                history.append({"role": role, "parts": [content]})

        last_prompt = messages[-1]["content"] if messages else ""
        return self._execute_generate_content(
            model_name=target_model,
            parts=[last_prompt],
            history=history,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ── Audio Input Processing ───────────────────────────────────────────

    def process_audio_input(
        self,
        audio_bytes: bytes,
        user_prompt: str = "Please listen to this audio and respond thoughtfully.",
        mime_type: str = "audio/wav",
        model_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Process incoming audio recording or audio file using Gemini.

        CRITICAL FIX FOR ERROR 1:
        - NEVER routes incoming audio to a Text-to-Speech (-tts) model.
        - Validates that model_name supports audio input.
        - Falls back to AUDIO_INPUT_MODEL (gemini-1.5-flash) if a TTS model was supplied.
        """
        target_model = model_name or config.AUDIO_INPUT_MODEL

        # STRICT GUARD: If model contains '-tts', immediately override with AUDIO_INPUT_MODEL
        if "-tts" in target_model.lower() or "preview-tts" in target_model.lower():
            logger.warning(
                f"Modality Guard: '{target_model}' is a Text-to-Speech generation model. "
                f"Redirecting incoming audio input to '{config.AUDIO_INPUT_MODEL}'."
            )
            target_model = config.AUDIO_INPUT_MODEL

        audio_part = {"mime_type": mime_type, "data": audio_bytes}
        parts = [user_prompt, audio_part]

        return self._execute_generate_content(
            model_name=target_model,
            parts=parts,
            system_instruction="You are an empathetic, helpful AI assistant. Transcribe and respond to the user's speech accurately.",
        )

    # ── Image Understanding (Vision) ─────────────────────────────────────

    def analyze_image(
        self,
        image_bytes: bytes,
        user_prompt: str = "Describe this image in detail.",
        model_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Analyze an image using Gemini Multimodal Vision."""
        target_model = model_name or config.IMAGE_MODEL
        if "-tts" in target_model.lower():
            target_model = config.IMAGE_MODEL

        img = Image.open(io.BytesIO(image_bytes))
        parts = [user_prompt, img]

        return self._execute_generate_content(
            model_name=target_model,
            parts=parts,
            system_instruction="You are a helpful, precise visual reasoning assistant.",
        )

    # ── Image Generation (Google Imagen 3) ────────────────────────────────

    def generate_image(
        self,
        prompt: str,
        model_name: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        """
        Generate an image using Google Imagen 3 (imagen-3.0-generate-002).
        Returns: (png_bytes, model_used)
        """
        from app.utils.gemini_client import generate_image_with_gemini, GeminiImageGenUnsupportedError
        target_model = model_name or config.IMAGEN_MODEL
        try:
            img = generate_image_with_gemini(api_key=self.api_key, prompt=prompt)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue(), target_model
        except GeminiImageGenUnsupportedError:
            raise
        except Exception as e:
            raise GeminiServiceError(f"Image generation failed: {e}") from e

    # ── Internal Execution with Quota Detection & Pacing ─────────────────

    def _execute_generate_content(
        self,
        model_name: str,
        parts: List[Any],
        history: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Tuple[str, str]:
        """
        Execute call with retry on transient rate limits, quota detection, and error mapping.
        """
        # Guard: Check if daily quota is already marked as active
        if quota_manager.is_daily_quota_active():
            eta_display = quota_manager.get_reset_display()
            raise GeminiDailyQuotaExhaustedError(
                f"Gemini daily quota is currently exhausted ({eta_display}). Switching to local fallback.",
                retry_after=quota_manager.get_remaining_seconds(),
            )

        genai = self._get_genai()
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Build fallback candidate list in case primary model returns 404 or modality error
        has_audio = any(
            (isinstance(p, dict) and str(p.get("mime_type", "")).startswith("audio/"))
            for p in parts
        )

        candidates = [model_name]
        default_candidates = [config.AUDIO_INPUT_MODEL, "gemini-1.5-flash", "gemini-2.0-flash"] if has_audio else [config.TEXT_MODEL, "gemini-1.5-flash", "gemini-2.0-flash"]
        for c in default_candidates:
            if c not in candidates and "-tts" not in c.lower():
                candidates.append(c)

        last_error = None
        max_retries = config.GEMINI_MAX_RETRIES

        for current_model in candidates:
            # Skip any TTS model if audio is present
            if has_audio and "-tts" in current_model.lower():
                continue

            attempt = 0
            while attempt <= max_retries:
                try:
                    model = genai.GenerativeModel(
                        model_name=current_model,
                        system_instruction=system_instruction or "",
                        generation_config=generation_config,
                    )
                    chat = model.start_chat(history=history or [])
                    response = chat.send_message(parts)

                    quota_manager.set_active_provider("gemini")
                    return response.text.strip(), current_model

                except Exception as exc:
                    last_error = exc
                    err_str = str(exc).lower()

                    # 1. Modality error: audio sent to incompatible model
                    if "audio input modality is not enabled" in err_str:
                        logger.warning(
                            f"Model '{current_model}' does not support audio input modality. "
                            "Switching to next candidate model..."
                        )
                        break  # Advance to next candidate model

                    # 2. Authentication failure (401)
                    if "401" in err_str or "unauthenticated" in err_str or "invalid api key" in err_str:
                        raise GeminiAuthFailureError(
                            "Invalid Gemini API key. Please check GEMINI_API_KEY in .env."
                        ) from exc

                    # 3. Model not found (404) — advance to next candidate
                    if "404" in err_str or "not found" in err_str:
                        logger.warning(f"Model '{current_model}' not found. Trying next candidate...")
                        break

                    # 4. Quota exhaustion / Rate limiting (429)
                    if is_rate_limit_error(exc):
                        retry_after = parse_retry_after(exc)
                        is_daily = is_daily_quota_error(exc)

                        if is_daily or (retry_after is not None and retry_after >= 3600):
                            # Daily free-tier limit exceeded!
                            quota_manager.mark_daily_quota(
                                retry_after_seconds=retry_after,
                                error_message=str(exc),
                            )
                            raise GeminiDailyQuotaExhaustedError(
                                "Gemini daily free-tier quota (20 requests/day) reached.",
                                retry_after=retry_after,
                            ) from exc

                        # Transient RPM rate limit: bounded backoff retry
                        attempt += 1
                        if attempt > max_retries:
                            logger.warning(f"Gemini RPM retries exhausted for '{current_model}'.")
                            raise GeminiRateLimitExhaustedError(
                                "Gemini rate limit exceeded. Please wait a moment and try again.",
                                retry_after=retry_after or 60.0,
                            ) from exc

                        sleep_time = retry_after if retry_after else (2.0 ** attempt + random.uniform(0.5, 2.0))
                        logger.info(f"Gemini RPM limit on '{current_model}'. Waiting {sleep_time:.1f}s before retry #{attempt}...")
                        time.sleep(sleep_time)
                        continue

                    # 5. Other unhandled errors
                    logger.error(f"Gemini API error on '{current_model}': {exc}")
                    raise GeminiServiceError(f"Gemini API error: {exc}") from exc

        raise GeminiServiceError(f"All Gemini candidate models failed: {last_error}")

