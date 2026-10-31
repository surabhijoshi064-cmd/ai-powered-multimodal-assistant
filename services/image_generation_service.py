"""
Dedicated Image Generation Service for Nexus AI.

Features:
  - Text-to-Image creation with Google Imagen & ComfyUI local engine.
  - Multi-aspect ratio support (1:1, 16:9, 9:16, 4:3, 3:4).
  - Rich style presets (Cinematic, Photorealistic, Anime, Cyberpunk, 3D Render, Logo Design, etc.).
  - Automatic fallback: Routes to ComfyUI when Gemini is unavailable or quota-limited.
  - Progress and metadata reporting.
"""
import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

import config
from services.quota_manager import quota_manager
from services.image_provider import GeminiImageProvider, ComfyUIImageProvider
from services.image_storage_service import image_storage_service

logger = logging.getLogger(__name__)

STYLE_PRESETS: Dict[str, str] = {
    "None": "",
    "Cinematic": "cinematic lighting, dramatic atmosphere, 8k resolution, photorealistic, shallow depth of field, anamorphic",
    "Photorealistic": "photorealistic, hyperrealistic, highly detailed, 35mm lens, natural lighting, sharp focus",
    "Anime / Manga": "modern anime art style, vibrant colors, clean linework, Makoto Shinkai aesthetic, studio quality",
    "Cyberpunk": "cyberpunk aesthetic, neon lighting, volumetric fog, futuristic city glow, high tech, dark synthwave",
    "3D Render": "octane render, Unreal Engine 5, ray tracing, subsurface scattering, stylized 3D character, soft lighting",
    "Business Logo": "minimalist modern vector logo, clean typography, flat design, white background, corporate branding",
    "Fantasy Digital Art": "digital concept art, epic fantasy, intricate details, ArtStation trending, mystical atmosphere",
    "Oil Painting": "classic oil painting, textured brushstrokes, rich impasto, Rembrandt lighting, museum masterpiece",
    "Watercolor": "delicate watercolor painting, soft pigment bleed, paper texture, pastel tones, artistic splash",
}

ASPECT_RATIOS: List[str] = ["1:1", "16:9", "9:16", "4:3", "3:4"]


@dataclass
class GenerationResult:
    """Outcome of an image generation job."""
    success: bool
    image_bytes: Optional[bytes]
    message: str
    provider: str               # "gemini" | "comfyui" | "none"
    model_used: str
    aspect_ratio: str
    style_applied: str
    prompt: str


class ImageGenerationService:
    """Manages Text-to-Image generation across cloud and local engines."""

    def __init__(self, gemini_provider=None, comfy_provider=None):
        self.gemini_provider = gemini_provider or GeminiImageProvider()
        self.comfy_provider = comfy_provider or ComfyUIImageProvider()

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        style: str = "Cinematic",
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        preferred_engine: str = "auto",  # "auto" | "gemini" | "comfyui"
    ) -> GenerationResult:
        """
        Execute a text-to-image generation job with intelligent routing & failover.
        """
        clean_prompt = prompt.strip()
        style_prompt = STYLE_PRESETS.get(style, "")
        augmented_prompt = f"{clean_prompt}, {style_prompt}".strip(" ,") if style_prompt else clean_prompt

        # ── 1. If user specifically requested ComfyUI ──
        if preferred_engine == "comfyui":
            if self.comfy_provider.is_available():
                try:
                    img_bytes, msg, model = self.comfy_provider.generate_image(
                        prompt=augmented_prompt,
                        aspect_ratio=aspect_ratio,
                        negative_prompt=negative_prompt,
                        seed=seed,
                    )
                    return GenerationResult(
                        success=img_bytes is not None,
                        image_bytes=img_bytes,
                        message=msg,
                        provider="comfyui",
                        model_used=model,
                        aspect_ratio=aspect_ratio,
                        style_applied=style,
                        prompt=clean_prompt,
                    )
                except Exception as e:
                    logger.warning(f"ComfyUI generation failed: {e}")
            return GenerationResult(
                success=False,
                image_bytes=None,
                message="ComfyUI local server is offline or unreachable at http://127.0.0.1:8188.",
                provider="none",
                model_used="none",
                aspect_ratio=aspect_ratio,
                style_applied=style,
                prompt=clean_prompt,
            )

        # ── 2. Primary Engine: Google Gemini Image API ──
        if self.gemini_provider.is_available():
            try:
                img_bytes, msg, model = self.gemini_provider.generate_image(
                    prompt=augmented_prompt,
                    aspect_ratio=aspect_ratio,
                    negative_prompt=negative_prompt,
                    seed=seed,
                )
                if img_bytes:
                    image_storage_service.save_generated_image(
                        image_bytes=img_bytes,
                        prompt=clean_prompt,
                        provider="gemini",
                        model=model,
                        aspect_ratio=aspect_ratio,
                        style=style,
                    )
                return GenerationResult(
                    success=img_bytes is not None,
                    image_bytes=img_bytes,
                    message=msg,
                    provider="gemini",
                    model_used=model,
                    aspect_ratio=aspect_ratio,
                    style_applied=style,
                    prompt=clean_prompt,
                )
            except Exception as e:
                logger.warning(f"Gemini image generation failed ({e}). Checking ComfyUI fallback...")

        # ── 3. Fallback Engine: ComfyUI Local Engine ──
        if self.comfy_provider.is_available():
            try:
                img_bytes, msg, model = self.comfy_provider.generate_image(
                    prompt=augmented_prompt,
                    aspect_ratio=aspect_ratio,
                    negative_prompt=negative_prompt,
                    seed=seed,
                )
                if img_bytes:
                    image_storage_service.save_generated_image(
                        image_bytes=img_bytes,
                        prompt=clean_prompt,
                        provider="comfyui",
                        model=model,
                        aspect_ratio=aspect_ratio,
                        style=style,
                    )
                prefix = "*(⚠️ Cloud Gemini quota reached — automatically switched to local ComfyUI engine)*\n\n"
                return GenerationResult(
                    success=True,
                    image_bytes=img_bytes,
                    message=prefix + msg,
                    provider="comfyui",
                    model_used=model,
                    aspect_ratio=aspect_ratio,
                    style_applied=style,
                    prompt=clean_prompt,
                )
            except Exception as e:
                logger.warning(f"ComfyUI fallback failed: {e}")

        # ── 4. Both engines unavailable ──
        eta = quota_manager.get_reset_display() if not quota_manager.is_gemini_available() else "soon"
        err_msg = (
            "⚠️ **Image Generation Unavailable**\n\n"
            f"1. **Google Gemini:** Quota exhausted or key not configured. Reset in ~{eta}.\n"
            "2. **ComfyUI Local Engine:** Server offline (checked `http://127.0.0.1:8188`).\n\n"
            "**To enable local image generation:**\n"
            "- Start ComfyUI locally: `run_nvidia_gpu.bat` (or your platform launcher)\n"
            "- Ensure a Stable Diffusion checkpoint (e.g. `v1-5-pruned-emaonly.safetensors` or SDXL) is in `ComfyUI/models/checkpoints/`"
        )
        return GenerationResult(
            success=False,
            image_bytes=None,
            message=err_msg,
            provider="none",
            model_used="none",
            aspect_ratio=aspect_ratio,
            style_applied=style,
            prompt=clean_prompt,
        )

