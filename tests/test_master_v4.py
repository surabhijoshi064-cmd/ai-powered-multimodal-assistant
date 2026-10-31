"""
NEXUS AI — Master Prompt V4.0 Automated Test Suite.

Covers:
  - Task Intent Routing (Chat, Image Gen, Image Edit, Vision, Audio, TTS)
  - ComfyUI Service Lifecycle & Workflow Generation
  - Image Generation Service & Style Presets
  - Image Editing Service, Presets & Fallback Filters
  - Model Separation & Fallback Guardrails
"""
import io
import pytest
from PIL import Image

import config
from services.ai_router import ai_router, TaskIntent
from services.comfyui_service import ComfyUIService, ComfyUIOfflineError
from services.image_provider import GeminiImageProvider, ComfyUIImageProvider
from services.image_generation_service import ImageGenerationService, STYLE_PRESETS, ASPECT_RATIOS
from services.image_editing_service import ImageEditingService, EDIT_PRESETS


class TestNexusIntentRouting:
    """Validate that AIRouter correctly classifies intent without hardcoding."""

    def test_chat_intent_for_general_questions(self):
        intent = ai_router.classify_intent("Explain binary trees in simple terms")
        assert intent == TaskIntent.CHAT

    def test_image_gen_intent_prefixes(self):
        prompts = [
            "Generate a picture of a lion in a cosmic forest",
            "Create an image of a futuristic city with flying cars",
            "Draw a picture of a cyberpunk warrior",
            "/imagine an ancient Japanese temple in cherry blossom season",
            "Design a logo for an AI startup",
        ]
        for p in prompts:
            assert ai_router.classify_intent(p) == TaskIntent.IMAGE_GEN

    def test_image_edit_intent_when_image_is_attached(self):
        prompts = [
            "Make this image look cinematic and dramatic",
            "Remove the background from this picture",
            "Replace the background with a tropical beach",
            "Add sunglasses to the person in this image",
            "Convert this to anime style",
        ]
        for p in prompts:
            assert ai_router.classify_intent(p, has_image=True) == TaskIntent.IMAGE_EDIT

    def test_image_understanding_intent_when_image_is_attached(self):
        prompts = [
            "Describe this picture in detail",
            "What is in this image?",
            "Explain the architecture shown here",
        ]
        for p in prompts:
            assert ai_router.classify_intent(p, has_image=True) == TaskIntent.IMAGE_UNDERSTANDING

    def test_audio_input_intent_when_audio_attached(self):
        assert ai_router.classify_intent("Any text", has_audio=True) == TaskIntent.AUDIO_INPUT

    def test_tts_intent_for_speech_commands(self):
        assert ai_router.classify_intent("Read this out loud for me") == TaskIntent.TTS
        assert ai_router.classify_intent("Convert to speech: Hello world") == TaskIntent.TTS


class TestComfyUIService:
    """Validate ComfyUI client behavior and workflow generators."""

    def test_comfyui_is_available_returns_bool(self):
        svc = ComfyUIService(base_url="http://127.0.0.1:9999")  # Unused port
        assert svc.is_available() is False

    def test_txt2img_workflow_structure(self):
        svc = ComfyUIService()
        wf = svc._build_txt2img_workflow(
            prompt="Cyberpunk city",
            negative_prompt="blurry",
            width=768,
            height=512,
            steps=25,
            cfg=8.0,
            seed=12345,
            model_name="sd_v1-5.safetensors",
        )
        assert "3" in wf and wf["3"]["class_type"] == "KSampler"
        assert wf["3"]["inputs"]["steps"] == 25
        assert wf["3"]["inputs"]["cfg"] == 8.0
        assert wf["3"]["inputs"]["seed"] == 12345
        assert wf["5"]["inputs"]["width"] == 768
        assert wf["5"]["inputs"]["height"] == 512
        assert wf["6"]["inputs"]["text"] == "Cyberpunk city"

    def test_img2img_workflow_structure(self):
        svc = ComfyUIService()
        wf = svc._build_img2img_workflow(
            image_name="input.png",
            prompt="Make it anime",
            denoise=0.7,
            steps=20,
        )
        assert "1" in wf and wf["1"]["class_type"] == "LoadImage"
        assert wf["1"]["inputs"]["image"] == "input.png"
        assert wf["3"]["inputs"]["denoise"] == 0.7
        assert wf["6"]["inputs"]["text"] == "Make it anime"

    def test_offline_comfyui_raises_clean_error(self):
        svc = ComfyUIService(base_url="http://127.0.0.1:9999")
        with pytest.raises(ComfyUIOfflineError):
            svc.generate_image("A futuristic city")


class TestImageGenerationService:
    """Validate text-to-image studio service."""

    def test_style_presets_available(self):
        assert "Cinematic" in STYLE_PRESETS
        assert "Photorealistic" in STYLE_PRESETS
        assert "Anime / Manga" in STYLE_PRESETS
        assert "Cyberpunk" in STYLE_PRESETS
        assert "3D Render" in STYLE_PRESETS

    def test_aspect_ratios_supported(self):
        assert "1:1" in ASPECT_RATIOS
        assert "16:9" in ASPECT_RATIOS
        assert "9:16" in ASPECT_RATIOS

    def test_comfyui_offline_returns_clean_message_without_crash(self):
        svc = ImageGenerationService()
        res = svc.generate(
            prompt="A futuristic lion",
            preferred_engine="comfyui",
        )
        assert res.success is False
        assert "ComfyUI" in res.message


class TestImageEditingService:
    """Validate natural language reference-guided editing."""

    def test_edit_presets_available(self):
        assert "Background Removal" in EDIT_PRESETS
        assert "Tropical Beach Background" in EDIT_PRESETS
        assert "Anime Style Transfer" in EDIT_PRESETS
        assert "Cinematic Lighting" in EDIT_PRESETS

    def test_smart_visual_filter_fallback(self):
        # Create a small test image in memory
        img = Image.new("RGB", (64, 64), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        svc = ImageEditingService()
        filtered = svc._apply_smart_visual_filter(raw_bytes, "Convert to black and white")
        assert isinstance(filtered, bytes)
        assert len(filtered) > 0

        # Verify output is valid image
        out_img = Image.open(io.BytesIO(filtered))
        assert out_img.size == (64, 64)

    def test_edit_returns_edit_result_structure(self):
        img = Image.new("RGB", (32, 32), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        svc = ImageEditingService()
        result = svc.edit(
            image_bytes=raw_bytes,
            instruction="Make this cinematic",
        )
        assert result.original_bytes == raw_bytes
        assert result.instruction == "Make this cinematic"
        assert result.message is not None


class TestModelSeparationV4:
    """Ensure strict model role separation."""

    def test_audio_input_model_is_not_tts(self):
        assert "-tts" not in config.AUDIO_INPUT_MODEL.lower()

    def test_tts_model_is_designated_speech_model(self):
        assert "tts" in config.TTS_MODEL.lower()

    def test_ollama_is_never_treated_as_image_generation_engine(self):
        # Ollama provider does not implement image generation
        from services.ollama_service import OllamaService
        ollama = OllamaService()
        assert not hasattr(ollama, "generate_image")

