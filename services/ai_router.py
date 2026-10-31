"""
Centralized AI Provider & Task Router for Nexus AI.

Coordinates:
  - Task Intent Classification (Chat, Image Gen, Image Edit, Vision, Audio, TTS)
  - Cloud Provider: Google Gemini
  - Local AI Provider: Ollama
  - Local Image Engine: ComfyUI
  - Circuit breaker failover on HTTP 429 quota exhaustion
"""
import enum
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image

import config
from services.quota_manager import quota_manager
from services.gemini_service import (
    GeminiService,
    GeminiDailyQuotaExhaustedError,
    GeminiRateLimitExhaustedError,
    GeminiAuthFailureError,
    GeminiServiceError,
)
from services.ollama_service import (
    OllamaService,
    OllamaServerOfflineError,
    OllamaModelMissingError,
    OllamaUnsupportedModalityError,
    OllamaServiceError,
)
from services.text_service import TextService
from services.audio_input_service import AudioInputService
from services.image_service import ImageService
from services.image_generation_service import ImageGenerationService
from services.image_editing_service import ImageEditingService

logger = logging.getLogger(__name__)


class TaskIntent(enum.Enum):
    CHAT = "chat"
    IMAGE_GEN = "image_generation"
    IMAGE_EDIT = "image_editing"
    IMAGE_UNDERSTANDING = "image_understanding"
    AUDIO_INPUT = "audio_input"
    TTS = "text_to_speech"


@dataclass
class TurnResult:
    """Represents the output of a single user turn."""
    success: bool
    response_text: str
    provider: str                   # "gemini" | "ollama" | "comfyui" | "none"
    model_used: str
    intent: str = "chat"
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    generated_image_bytes: Optional[bytes] = None
    original_image_bytes: Optional[bytes] = None
    edited_image_bytes: Optional[bytes] = None
    error_type: Optional[str] = None
    error_title: Optional[str] = None
    error_message: Optional[str] = None


class AIRouter:
    """Master AI & Multimodal Task Router for Nexus AI."""

    def __init__(self):
        self.gemini_service = GeminiService()
        self.ollama_service = OllamaService()
        self.audio_service = AudioInputService(self.gemini_service, self.ollama_service)
        self.image_service = ImageService(self.gemini_service, self.ollama_service)
        self.generation_service = ImageGenerationService()
        self.editing_service = ImageEditingService()

    def classify_intent(
        self,
        prompt: str,
        has_image: bool = False,
        has_audio: bool = False,
    ) -> TaskIntent:
        """Classify user intent using semantic heuristics and modality signals."""
        lower_prompt = prompt.strip().lower()

        # 1. Audio input
        if has_audio:
            return TaskIntent.AUDIO_INPUT

        # 2. Image editing (requires an attached image and editing keywords)
        if has_image:
            edit_keywords = [
                "edit", "change", "remove background", "replace background", "add", "make it",
                "convert to", "filter", "enhance", "color", "cinematic", "anime", "style",
                "retouch", "sunglasses", "clothes", "background", "portrait"
            ]
            if any(k in lower_prompt for k in edit_keywords) and not lower_prompt.startswith(("what is", "describe", "explain")):
                return TaskIntent.IMAGE_EDIT
            return TaskIntent.IMAGE_UNDERSTANDING

        # 3. Image generation (explicit creation keywords)
        image_gen_prefixes = (
            "/image", "/imagine", "generate an image", "generate a picture", "create an image",
            "create a picture", "draw an image", "draw a picture", "design a logo", "make a picture",
            "make an image", "generate a photo", "generate a portrait", "paint a", "render a"
        )
        if lower_prompt.startswith(image_gen_prefixes) or "generate an image of" in lower_prompt or "create a picture of" in lower_prompt:
            return TaskIntent.IMAGE_GEN

        # 4. Text-to-speech request
        if lower_prompt.startswith(("read this", "speak this", "convert to speech", "read aloud", "say this")):
            return TaskIntent.TTS

        # 5. Default to Chat Assistant
        return TaskIntent.CHAT

    def route_turn(
        self,
        prompt: str,
        messages_history: List[Dict[str, str]],
        attached_image: Optional[Image.Image] = None,
        attached_image_bytes: Optional[bytes] = None,
        attached_audio_bytes: Optional[bytes] = None,
        attached_doc_text: Optional[str] = None,
        attached_doc_name: Optional[str] = None,
        requested_gemini_model: Optional[str] = None,
        requested_ollama_model: Optional[str] = None,
        aspect_ratio: str = "1:1",
        style: str = "Cinematic",
    ) -> TurnResult:
        """
        Execute an end-to-end multimodal conversation turn with intelligent routing.
        """
        # Resolve image bytes if PIL Image provided
        raw_image_bytes = attached_image_bytes
        if raw_image_bytes is None and attached_image is not None:
            import io
            buf = io.BytesIO()
            attached_image.save(buf, format="PNG")
            raw_image_bytes = buf.getvalue()

        intent = self.classify_intent(
            prompt=prompt,
            has_image=raw_image_bytes is not None,
            has_audio=attached_audio_bytes is not None,
        )

        # ── Route 1: Text-to-Image Generation ──
        if intent == TaskIntent.IMAGE_GEN:
            clean_prompt = prompt
            for prefix in (
                "/image", "/imagine", "generate an image of", "generate an image",
                "generate a picture of", "generate a picture", "create an image of",
                "create an image", "create a picture of", "draw a picture of",
                "draw an image of", "design a logo of", "make an image of"
            ):
                if clean_prompt.lower().startswith(prefix):
                    clean_prompt = clean_prompt[len(prefix):].strip(" :,-")
                    break

            gen_result = self.generation_service.generate(
                prompt=clean_prompt or prompt,
                aspect_ratio=aspect_ratio,
                style=style,
            )
            return TurnResult(
                success=gen_result.success,
                response_text=gen_result.message,
                provider=gen_result.provider,
                model_used=gen_result.model_used,
                intent=TaskIntent.IMAGE_GEN.value,
                generated_image_bytes=gen_result.image_bytes,
            )

        # ── Route 2: Image Editing Studio ──
        if intent == TaskIntent.IMAGE_EDIT and raw_image_bytes is not None:
            edit_result = self.editing_service.edit(
                image_bytes=raw_image_bytes,
                instruction=prompt,
                style=style,
            )
            return TurnResult(
                success=edit_result.success,
                response_text=edit_result.message,
                provider=edit_result.provider,
                model_used=edit_result.model_used,
                intent=TaskIntent.IMAGE_EDIT.value,
                original_image_bytes=raw_image_bytes,
                edited_image_bytes=edit_result.edited_bytes,
                generated_image_bytes=edit_result.edited_bytes,
            )

        # ── Route 3: Audio Input Processing ──
        if intent == TaskIntent.AUDIO_INPUT and attached_audio_bytes is not None:
            text, prov, model = self.audio_service.process_audio(
                audio_bytes=attached_audio_bytes,
                user_prompt=prompt or "Please listen to this voice prompt and respond thoughtfully.",
                mime_type="audio/wav",
            )
            return TurnResult(
                success=prov != "none",
                response_text=text,
                provider=prov,
                model_used=model,
                intent=TaskIntent.AUDIO_INPUT.value,
                is_fallback=(prov == "ollama"),
                fallback_reason="gemini_daily_quota" if prov == "ollama" else None,
            )

        # ── Route 4: Image Understanding (Vision Analysis) ──
        if intent == TaskIntent.IMAGE_UNDERSTANDING and raw_image_bytes is not None:
            text, prov, model = self.image_service.analyze_image(
                image_bytes=raw_image_bytes,
                user_prompt=prompt or "Describe this image in detail.",
            )
            return TurnResult(
                success=prov != "none",
                response_text=text,
                provider=prov,
                model_used=model,
                intent=TaskIntent.IMAGE_UNDERSTANDING.value,
                is_fallback=(prov == "ollama"),
                fallback_reason="gemini_daily_quota" if prov == "ollama" else None,
            )

        # ── Route 5: Standard Text & Document Chat ──
        formatted_messages = TextService.format_history(
            messages_history=messages_history,
            current_prompt=prompt,
            doc_text=attached_doc_text,
            doc_name=attached_doc_name,
        )

        # Primary: Gemini
        if quota_manager.is_gemini_available():
            try:
                reply, model = self.gemini_service.generate_text(
                    messages=formatted_messages,
                    system_instruction=TextService.build_system_prompt(),
                    model_name=requested_gemini_model or config.TEXT_MODEL,
                )
                return TurnResult(
                    success=True,
                    response_text=reply,
                    provider="gemini",
                    model_used=model,
                    intent=TaskIntent.CHAT.value,
                )
            except GeminiDailyQuotaExhaustedError:
                logger.warning("Gemini daily quota hit. Initiating automatic Ollama fallback...")
            except GeminiAuthFailureError as auth_err:
                return TurnResult(
                    success=False,
                    response_text=str(auth_err),
                    provider="gemini",
                    model_used="none",
                    intent=TaskIntent.CHAT.value,
                    error_type="auth_failure",
                    error_title="🔑 Gemini API Key Invalid",
                    error_message="Your Gemini API key was rejected by Google. Please verify your GEMINI_API_KEY in .env.",
                )
            except GeminiRateLimitExhaustedError as rpm_err:
                return TurnResult(
                    success=False,
                    response_text=str(rpm_err),
                    provider="gemini",
                    model_used="none",
                    intent=TaskIntent.CHAT.value,
                    error_type="rate_limit",
                    error_title="⏸️ Gemini Rate Limit Paused",
                    error_message=f"Gemini per-minute rate limit reached. Please wait ~{int(rpm_err.retry_after)}s before retrying.",
                )
            except GeminiServiceError as srv_err:
                logger.warning(f"Gemini error ({srv_err}). Testing Ollama fallback...")

        # Secondary: Ollama
        if config.OLLAMA_ENABLED:
            if not self.ollama_service.is_running():
                if quota_manager.is_daily_quota_active():
                    eta = quota_manager.get_reset_display()
                    msg = (
                        "⏳ **Gemini Daily Quota Exceeded & Local AI Offline**\n\n"
                        "Your Gemini free-tier daily quota has been reached.\n"
                        f"> Reset estimate: **{eta}**.\n\n"
                        "Nexus AI is configured to use local Ollama AI as an automatic fallback, "
                        "but the Ollama server is currently not running.\n\n"
                        "**To activate your local AI assistant:**\n"
                        "1. Open Windows PowerShell.\n"
                        "2. Run: `ollama serve`\n"
                        f"3. Ensure your model is downloaded: `ollama pull {config.OLLAMA_TEXT_MODEL}`"
                    )
                    return TurnResult(
                        success=False,
                        response_text=msg,
                        provider="none",
                        model_used="none",
                        intent=TaskIntent.CHAT.value,
                        error_type="ollama_offline",
                        error_title="🖥️ Ollama Local Server Offline",
                        error_message="Start Ollama with `ollama serve` to continue chatting offline.",
                    )
                else:
                    msg = (
                        "⚠️ **No AI Provider Available**\n\n"
                        "1. Gemini API is not configured or unavailable.\n"
                        "2. Local Ollama server is currently offline.\n\n"
                        "**To start chatting:**\n"
                        "- Add your `GEMINI_API_KEY` to `.env`\n"
                        "- Or run `ollama serve` to use your local AI assistant."
                    )
                    return TurnResult(
                        success=False,
                        response_text=msg,
                        provider="none",
                        model_used="none",
                        intent=TaskIntent.CHAT.value,
                        error_type="no_provider",
                        error_title="⚠️ No AI Provider Available",
                        error_message="Configure Gemini or start Ollama.",
                    )

            try:
                target_ollama = requested_ollama_model or config.OLLAMA_TEXT_MODEL
                reply, model = self.ollama_service.generate_text(
                    messages=formatted_messages,
                    system_instruction=TextService.build_system_prompt(),
                    model_name=target_ollama,
                )
                return TurnResult(
                    success=True,
                    response_text=reply,
                    provider="ollama",
                    model_used=model,
                    intent=TaskIntent.CHAT.value,
                    is_fallback=True,
                    fallback_reason="gemini_daily_quota",
                )
            except OllamaModelMissingError as mm_err:
                return TurnResult(
                    success=False,
                    response_text=str(mm_err),
                    provider="ollama",
                    model_used="none",
                    intent=TaskIntent.CHAT.value,
                    error_type="ollama_model_missing",
                    error_title="📦 Local Model Missing",
                    error_message=str(mm_err),
                )
            except Exception as ollama_err:
                return TurnResult(
                    success=False,
                    response_text=f"Local Ollama error: {ollama_err}",
                    provider="ollama",
                    model_used="none",
                    intent=TaskIntent.CHAT.value,
                    error_type="ollama_error",
                    error_title="⚠️ Local AI Error",
                    error_message=str(ollama_err),
                )

        return TurnResult(
            success=False,
            response_text=(
                "⚠️ **No AI Provider Available**\n\n"
                "1. Gemini API key is missing or its daily quota is exhausted.\n"
                "2. Ollama local fallback is disabled or offline.\n\n"
                "Please configure `GEMINI_API_KEY` in `.env` or set `OLLAMA_ENABLED=true` and run `ollama serve`."
            ),
            provider="none",
            model_used="none",
            intent=TaskIntent.CHAT.value,
            error_type="no_provider",
            error_title="⚠️ No AI Provider Available",
            error_message="Configure Gemini in .env or start Ollama.",
        )


# Singleton router
ai_router = AIRouter()
