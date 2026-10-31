"""
Image Understanding, Generation, and Editing Service for Nexus AI.

Coordinates:
  - Multimodal Vision Analysis (Gemini Vision & Ollama Vision)
  - Text-to-Image Generation (ImageGenerationService: Imagen 3 & ComfyUI)
  - Natural Language Image Editing (ImageEditingService: Reference-guided edits & ComfyUI)
"""
import logging
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

import config
from services.quota_manager import quota_manager
from services.image_generation_service import ImageGenerationService, GenerationResult
from services.image_editing_service import ImageEditingService, EditResult

logger = logging.getLogger(__name__)


class ImageService:
    """Master Image Service coordinating vision analysis, generation, and editing."""

    def __init__(self, gemini_service=None, ollama_service=None):
        self.gemini_service = gemini_service
        self.ollama_service = ollama_service
        self.generation_service = ImageGenerationService()
        self.editing_service = ImageEditingService()

    def _get_gemini_service(self):
        if self.gemini_service is None:
            from services.gemini_service import GeminiService
            self.gemini_service = GeminiService()
        return self.gemini_service

    def _get_ollama_service(self):
        if self.ollama_service is None:
            from services.ollama_service import OllamaService
            self.ollama_service = OllamaService()
        return self.ollama_service

    def analyze_image(
        self,
        image_bytes: bytes,
        user_prompt: str = "Describe this image in detail.",
    ) -> Tuple[str, str, str]:
        """
        Analyze an image for vision understanding.
        Returns: (analysis_text, provider_used, model_used)
        """
        # Primary: Gemini Vision
        if quota_manager.is_gemini_available():
            try:
                gemini = self._get_gemini_service()
                text, model = gemini.analyze_image(
                    image_bytes=image_bytes,
                    user_prompt=user_prompt,
                    model_name=config.IMAGE_MODEL,
                )
                return text, "gemini", model
            except Exception as e:
                logger.warning(f"Gemini image analysis failed ({e}). Attempting local fallback...")

        # Fallback: Local Ollama Vision
        ollama = self._get_ollama_service()
        if ollama.is_running():
            try:
                text, model = ollama.analyze_image(
                    image_bytes=image_bytes,
                    user_prompt=user_prompt,
                    model_name=config.OLLAMA_VISION_MODEL,
                )
                prefix = f"*(👁️ Image analyzed by local vision model: `{model}`)*\n\n"
                return prefix + text, "ollama", model
            except Exception as e:
                return (
                    f"⚠️ Image analysis unavailable: {e}\n\n"
                    "> *Note: Gemini quota is exhausted, and local vision requires a multimodal model "
                    "such as `moondream` or `llava:7b` (`ollama pull moondream`).*",
                    "ollama",
                    config.OLLAMA_VISION_MODEL,
                )

        return (
            "⚠️ Image analysis unavailable: Gemini daily quota is exhausted and local Ollama server is offline.",
            "none",
            "none",
        )

    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        style: str = "Cinematic",
    ) -> Tuple[Optional[bytes], str, str]:
        """
        Generate an image using the generation service.
        Returns: (png_bytes_or_None, message_text, provider_used)
        """
        result: GenerationResult = self.generation_service.generate(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            style=style,
        )
        return result.image_bytes, result.message, result.provider

    def edit_image(
        self,
        image_bytes: bytes,
        instruction: str,
        style: Optional[str] = None,
        preset_name: Optional[str] = None,
    ) -> Tuple[Optional[bytes], str, str]:
        """
        Edit an existing reference image using natural language.
        Returns: (edited_png_bytes_or_None, message_text, provider_used)
        """
        result: EditResult = self.editing_service.edit(
            image_bytes=image_bytes,
            instruction=instruction,
            style=style,
            preset_name=preset_name,
        )
        return result.edited_bytes, result.message, result.provider
