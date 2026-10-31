"""
Image Provider Abstraction for Nexus AI.

Defines unified interface for:
  - GeminiImageProvider (Google Imagen 3 & Gemini Multimodal Generation)
  - ComfyUIImageProvider (Local ComfyUI SD1.5/SDXL Workflows)
"""
import io
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

import config
from services.quota_manager import quota_manager

logger = logging.getLogger(__name__)


class BaseImageProvider(ABC):
    """Abstract interface for image generation and editing engines."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and ready to accept jobs."""
        pass

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        style: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Optional[bytes], str, str]:
        """
        Generate an image from text.
        Returns: (png_bytes_or_None, message_text, model_used)
        """
        pass

    @abstractmethod
    def edit_image(
        self,
        image_bytes: bytes,
        instruction: str,
        style: Optional[str] = None,
        denoise_strength: float = 0.65,
    ) -> Tuple[Optional[bytes], str, str]:
        """
        Edit an existing reference image using natural language instructions.
        Returns: (png_bytes_or_None, message_text, model_used)
        """
        pass


class GeminiImageProvider(BaseImageProvider):
    """Primary Cloud Image Engine using Google Gemini / Imagen."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.GEMINI_API_KEY

    def is_available(self) -> bool:
        return bool(self.api_key) and quota_manager.is_gemini_available()

    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        style: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Optional[bytes], str, str]:
        if not self.is_available():
            eta = quota_manager.get_reset_display()
            return None, f"Gemini Image API is currently unavailable or quota exhausted. Reset in ~{eta}.", "none"

        # Apply style enhancement to prompt if provided
        final_prompt = prompt
        if style and style.lower() != "none" and style.lower() != "default":
            final_prompt = f"{prompt}, {style} style, masterpiece, high resolution, detailed"

        try:
            from app.utils.gemini_client import generate_image_with_gemini
            img_obj = generate_image_with_gemini(
                api_key=self.api_key,
                prompt=final_prompt,
                aspect_ratio=aspect_ratio,
                model_name=config.IMAGEN_MODEL,
            )
            buf = io.BytesIO()
            img_obj.save(buf, format="PNG")
            return buf.getvalue(), "Image generated successfully via Google Imagen.", config.IMAGEN_MODEL
        except Exception as e:
            logger.warning(f"Gemini image generation error: {e}")
            raise e

    def edit_image(
        self,
        image_bytes: bytes,
        instruction: str,
        style: Optional[str] = None,
        denoise_strength: float = 0.65,
    ) -> Tuple[Optional[bytes], str, str]:
        """
        Reference-guided image editing using Gemini Vision + Image Synthesis.
        1. Analyzes the source image to understand content and structure.
        2. Applies the requested natural language transformation.
        """
        if not self.is_available():
            eta = quota_manager.get_reset_display()
            return None, f"Gemini Image API is currently unavailable or quota exhausted. Reset in ~{eta}.", "none"

        try:
            # 1. Vision stage: describe original image with edit requirements
            from services.gemini_service import GeminiService
            gemini_svc = GeminiService()
            analysis_prompt = (
                f"You are an expert visual director. The user wants to edit this image with the instruction: '{instruction}'.\n"
                "Describe the desired final edited image in vivid detail, preserving the subject and main composition of this "
                "original image while fully incorporating the edit instruction. "
                "Output ONLY a single prompt suitable for image generation, under 100 words."
            )
            synthesis_prompt, _ = gemini_svc.analyze_image(
                image_bytes=image_bytes,
                user_prompt=analysis_prompt,
                model_name=config.IMAGE_MODEL,
            )

            # 2. Add style if specified
            if style and style.lower() != "none" and style.lower() != "default":
                synthesis_prompt = f"{synthesis_prompt}, {style} aesthetic, highly detailed"

            # 3. Generate the edited image using Imagen
            from app.utils.gemini_client import generate_image_with_gemini
            edited_img = generate_image_with_gemini(
                api_key=self.api_key,
                prompt=synthesis_prompt,
                aspect_ratio="1:1",
                model_name=config.IMAGEN_MODEL,
            )

            buf = io.BytesIO()
            edited_img.save(buf, format="PNG")
            return (
                buf.getvalue(),
                f"Image edited successfully according to: '{instruction}'.",
                config.IMAGEN_MODEL,
            )
        except Exception as e:
            logger.warning(f"Gemini image editing error: {e}")
            raise e


class ComfyUIImageProvider(BaseImageProvider):
    """Secondary Local Image Engine using ComfyUI."""

    def __init__(self, base_url: Optional[str] = None):
        from services.comfyui_service import ComfyUIService
        self.comfy = ComfyUIService(base_url=base_url)

    def is_available(self) -> bool:
        return self.comfy.is_available()

    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        style: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Optional[bytes], str, str]:
        if not self.is_available():
            return None, "ComfyUI local server is offline at http://127.0.0.1:8188.", "none"

        # Aspect ratio to width/height
        dims = {"1:1": (512, 512), "16:9": (768, 432), "9:16": (432, 768), "4:3": (640, 480), "3:4": (480, 640)}
        width, height = dims.get(aspect_ratio, (512, 512))

        final_prompt = prompt
        if style and style.lower() != "none":
            final_prompt = f"{prompt}, {style} style, masterpiece, best quality"

        try:
            img_bytes, model = self.comfy.generate_image(
                prompt=final_prompt,
                negative_prompt=negative_prompt or "low quality, blurry, distorted",
                width=width,
                height=height,
                seed=seed,
            )
            return img_bytes, f"Generated locally using ComfyUI ({model}).", f"comfyui:{model}"
        except Exception as e:
            logger.warning(f"ComfyUI generate error: {e}")
            raise e

    def edit_image(
        self,
        image_bytes: bytes,
        instruction: str,
        style: Optional[str] = None,
        denoise_strength: float = 0.65,
    ) -> Tuple[Optional[bytes], str, str]:
        if not self.is_available():
            return None, "ComfyUI local server is offline at http://127.0.0.1:8188.", "none"

        final_prompt = instruction
        if style and style.lower() != "none":
            final_prompt = f"{instruction}, {style} style, high quality"

        try:
            img_bytes, model = self.comfy.edit_image(
                image_bytes=image_bytes,
                prompt=final_prompt,
                denoise=denoise_strength,
            )
            return img_bytes, f"Edited locally via ComfyUI img2img ({model}).", f"comfyui:{model}"
        except Exception as e:
            logger.warning(f"ComfyUI edit error: {e}")
            raise e

