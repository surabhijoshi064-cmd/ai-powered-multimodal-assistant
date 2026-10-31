"""
Dedicated AI Image Editing Studio Service for Nexus AI.

Features:
  - Natural language image editing with visual reference preservation.
  - Quick Tool Presets:
      • Background Removal (transparent or studio background)
      • Background Replacement (beach, cyberpunk, studio, nature, neon grid)
      • Style Transfer (Anime, Cyberpunk, Watercolor, Sketch, Oil Painting)
      • Color Grading & Lighting Enhancement
      • Object Addition & Removal
      • Portrait & Face Beautification
  - Before / After visual comparison data generation.
  - Cloud (Gemini) + Local (ComfyUI) engine routing.
"""
import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

import config
from services.quota_manager import quota_manager
from services.image_provider import GeminiImageProvider, ComfyUIImageProvider
from services.image_storage_service import image_storage_service

logger = logging.getLogger(__name__)

EDIT_PRESETS: Dict[str, str] = {
    "Background Removal": "Remove the background completely, keep only the central subject crisp and isolated with a transparent or pure white studio backdrop.",
    "Tropical Beach Background": "Keep the main subject exactly as is, but replace the background with a sunlit tropical beach with turquoise water and palm trees.",
    "Cyberpunk City Background": "Keep the main subject, replace the background with a futuristic cyberpunk city with neon holograms, rain reflections, and glowing skyscrapers.",
    "Minimal Studio Background": "Isolate the subject in a clean, high-end commercial photo studio with soft warm rim lighting and smooth gradient backdrop.",
    "Anime Style Transfer": "Convert the image into a high-end modern anime art style, with expressive eyes, vivid coloring, and clean stylized linework while keeping identity.",
    "Cyberpunk Style Transfer": "Transform the entire image into a high-tech cyberpunk aesthetic with neon accents, synthetic glow, dark tones, and cybernetic details.",
    "Cinematic Lighting": "Enhance lighting to dramatic cinematic style: rich contrast, warm key light, teal shadows, and subtle lens flare.",
    "Watercolor Painting": "Recreate this image as a delicate, artistic watercolor painting with organic brushstrokes and paper texture.",
    "Add Sunglasses": "Add stylish modern sunglasses onto the face of the person in the image, blending naturally with face lighting.",
    "Portrait Enhancement": "Professional portrait retouching: smooth natural skin tones, brighten eyes, enhance hair shine, and soften background focus.",
}


@dataclass
class EditResult:
    """Outcome of an image editing job."""
    success: bool
    original_bytes: bytes
    edited_bytes: Optional[bytes]
    instruction: str
    message: str
    provider: str               # "gemini" | "comfyui" | "pillow" | "none"
    model_used: str
    preset_used: Optional[str] = None


class ImageEditingService:
    """Coordinates image editing workflows across Gemini and ComfyUI."""

    def __init__(self, gemini_provider=None, comfy_provider=None):
        self.gemini_provider = gemini_provider or GeminiImageProvider()
        self.comfy_provider = comfy_provider or ComfyUIImageProvider()

    def edit(
        self,
        image_bytes: bytes,
        instruction: str,
        style: Optional[str] = None,
        preset_name: Optional[str] = None,
        denoise_strength: float = 0.65,
        preferred_engine: str = "auto",  # "auto" | "gemini" | "comfyui"
    ) -> EditResult:
        """
        Execute natural language image edit using original image as reference.
        """
        final_instruction = instruction
        if preset_name and preset_name in EDIT_PRESETS:
            final_instruction = f"{EDIT_PRESETS[preset_name]} {instruction}".strip()

        # ── 1. If user specifically requested ComfyUI ──
        if preferred_engine == "comfyui":
            if self.comfy_provider.is_available():
                try:
                    out_bytes, msg, model = self.comfy_provider.edit_image(
                        image_bytes=image_bytes,
                        instruction=final_instruction,
                        style=style,
                        denoise_strength=denoise_strength,
                    )
                    return EditResult(
                        success=out_bytes is not None,
                        original_bytes=image_bytes,
                        edited_bytes=out_bytes,
                        instruction=final_instruction,
                        message=msg,
                        provider="comfyui",
                        model_used=model,
                        preset_used=preset_name,
                    )
                except Exception as e:
                    logger.warning(f"ComfyUI image editing failed: {e}")
            return EditResult(
                success=False,
                original_bytes=image_bytes,
                edited_bytes=None,
                instruction=final_instruction,
                message="ComfyUI is offline at http://127.0.0.1:8188.",
                provider="none",
                model_used="none",
                preset_used=preset_name,
            )

        # ── 2. Primary Engine: Google Gemini ──
        if self.gemini_provider.is_available():
            try:
                out_bytes, msg, model = self.gemini_provider.edit_image(
                    image_bytes=image_bytes,
                    instruction=final_instruction,
                    style=style,
                    denoise_strength=denoise_strength,
                )
                if out_bytes:
                    image_storage_service.save_edited_image(
                        edited_bytes=out_bytes,
                        original_bytes=image_bytes,
                        instruction=final_instruction,
                        provider="gemini",
                        model=model,
                        preset=preset_name,
                    )
                return EditResult(
                    success=out_bytes is not None,
                    original_bytes=image_bytes,
                    edited_bytes=out_bytes,
                    instruction=final_instruction,
                    message=msg,
                    provider="gemini",
                    model_used=model,
                    preset_used=preset_name,
                )
            except Exception as e:
                logger.warning(f"Gemini image editing failed ({e}). Checking ComfyUI fallback...")

        # ── 3. Fallback Engine: ComfyUI Local Engine ──
        if self.comfy_provider.is_available():
            try:
                out_bytes, msg, model = self.comfy_provider.edit_image(
                    image_bytes=image_bytes,
                    instruction=final_instruction,
                    style=style,
                    denoise_strength=denoise_strength,
                )
                if out_bytes:
                    image_storage_service.save_edited_image(
                        edited_bytes=out_bytes,
                        original_bytes=image_bytes,
                        instruction=final_instruction,
                        provider="comfyui",
                        model=model,
                        preset=preset_name,
                    )
                prefix = "*(⚠️ Cloud Gemini unavailable — automatically switched to local ComfyUI engine)*\n\n"
                return EditResult(
                    success=True,
                    original_bytes=image_bytes,
                    edited_bytes=out_bytes,
                    instruction=final_instruction,
                    message=prefix + msg,
                    provider="comfyui",
                    model_used=model,
                    preset_used=preset_name,
                )
            except Exception as e:
                logger.warning(f"ComfyUI edit fallback failed: {e}")

        # ── 4. Client-side Smart Visual Filter Fallback ──
        # If cloud quota is exhausted and local ComfyUI is offline, perform deterministic PIL visual enhancement
        try:
            enhanced_bytes = self._apply_smart_visual_filter(image_bytes, final_instruction)
            return EditResult(
                success=True,
                original_bytes=image_bytes,
                edited_bytes=enhanced_bytes,
                instruction=final_instruction,
                message="*(⚠️ Advanced diffusion engines offline — applied client-side visual enhancement filter)*",
                provider="pillow_fallback",
                model_used="local_image_filters",
                preset_used=preset_name,
            )
        except Exception as e:
            logger.debug(f"Pillow filter fallback failed: {e}")

        eta = quota_manager.get_reset_display() if not quota_manager.is_gemini_available() else "soon"
        err_msg = (
            "⚠️ **Image Editing Studio Unavailable**\n\n"
            f"1. **Gemini Image Engine:** Quota limit reached. Reset in ~{eta}.\n"
            "2. **ComfyUI Local Engine:** Offline at `http://127.0.0.1:8188`.\n\n"
            "> *To edit images locally: start ComfyUI with an img2img capable model.*"
        )
        return EditResult(
            success=False,
            original_bytes=image_bytes,
            edited_bytes=None,
            instruction=final_instruction,
            message=err_msg,
            provider="none",
            model_used="none",
            preset_used=preset_name,
        )

    def _apply_smart_visual_filter(self, image_bytes: bytes, instruction: str) -> bytes:
        """Deterministic PIL image enhancements as graceful offline fallback."""
        with io.BytesIO(image_bytes) as in_buf:
            img = Image.open(in_buf).convert("RGB")

        inst_lower = instruction.lower()

        if "black and white" in inst_lower or "grayscale" in inst_lower or "sketch" in inst_lower:
            img = ImageOps.grayscale(img).convert("RGB")
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.4)
        elif "cinematic" in inst_lower or "lighting" in inst_lower:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.25)
            color_enhancer = ImageEnhance.Color(img)
            img = color_enhancer.enhance(1.15)
            sharp_enhancer = ImageEnhance.Sharpness(img)
            img = sharp_enhancer.enhance(1.3)
        elif "enhance" in inst_lower or "portrait" in inst_lower or "quality" in inst_lower:
            sharp_enhancer = ImageEnhance.Sharpness(img)
            img = sharp_enhancer.enhance(1.5)
            bright_enhancer = ImageEnhance.Brightness(img)
            img = bright_enhancer.enhance(1.05)
        elif "cyberpunk" in inst_lower or "neon" in inst_lower:
            # Shift colors towards cyan & violet
            r, g, b = img.split()
            r = r.point(lambda i: min(255, int(i * 1.2)))
            b = b.point(lambda i: min(255, int(i * 1.4)))
            img = Image.merge("RGB", (r, g, b))
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
        else:
            # Default gentle enhancement
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.1)

        out_buf = io.BytesIO()
        img.save(out_buf, format="PNG")
        return out_buf.getvalue()

