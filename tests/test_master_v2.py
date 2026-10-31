"""
Comprehensive Test Suite for Master Prompt V2.0 Requirements.

Covers:
  - TEST 1: Text Generation (Gemini & Ollama)
  - TEST 2: Audio Input Processing & Separation from TTS (Verifies fix for Error 1: 400 Audio input modality not enabled)
  - TEST 3: Audio Output (TTS Service & clean speech formatting)
  - TEST 4: Image Understanding & Analysis
  - TEST 5: Gemini Quota Exceeded (Detection, QuotaManager lock, stop retries, switch to Ollama)
  - TEST 6: Ollama Unavailable (Helpful notification explaining `ollama serve`)
  - TEST 7: Both Providers Unavailable (Graceful error handling, no application crash)
  - TEST 8: Provider Recovery (Safe recovery after cooldown expires)
  - TEST 9: Multimodal Capability Enforcement (Verifies no false capability claims)
"""
import io
import time
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

import config
from services.quota_manager import quota_manager, QuotaManager
from services.gemini_service import (
    GeminiService,
    GeminiDailyQuotaExhaustedError,
    GeminiRateLimitExhaustedError,
)
from services.ollama_service import (
    OllamaService,
    OllamaServerOfflineError,
    OllamaUnsupportedModalityError,
)
from services.audio_input_service import AudioInputService
from services.tts_service import TTSService
from services.image_service import ImageService
from services.ai_router import AIRouter


@pytest.fixture(autouse=True)
def clean_quota_state():
    """Ensure quota manager state is clean before each test."""
    quota_manager.clear_daily_quota()
    yield
    quota_manager.clear_daily_quota()


# ---------------------------------------------------------------------------
# TEST 1: Text Generation
# ---------------------------------------------------------------------------
class TestTextGeneration:
    @patch("services.gemini_service.GeminiService._get_genai")
    def test_gemini_text_generation_success(self, mock_get_genai):
        mock_genai = MagicMock()
        mock_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock(text="Peace begins with a conscious breath.")
        mock_chat.send_message.return_value = mock_response
        mock_model.start_chat.return_value = mock_chat
        mock_genai.GenerativeModel.return_value = mock_model
        mock_get_genai.return_value = mock_genai

        service = GeminiService(api_key="valid-key")
        reply, used_model = service.generate_text([{"role": "user", "content": "Hello"}])
        assert "Peace begins" in reply
        assert used_model == config.TEXT_MODEL


# ---------------------------------------------------------------------------
# TEST 2: Audio Input & Separation from TTS (Fix for Error 1)
# ---------------------------------------------------------------------------
class TestAudioInputFix:
    def test_audio_input_never_uses_tts_model(self):
        """
        Verify that passing a -tts model (like gemini-2.5-flash-preview-tts)
        to process_audio_input is automatically overridden by AUDIO_INPUT_MODEL.
        """
        service = GeminiService(api_key="valid-key")
        with patch.object(service, "_execute_generate_content") as mock_exec:
            mock_exec.return_value = ("Audio processed successfully", "gemini-1.5-flash")

            # Intentionally pass the TTS preview model that caused the 400 error
            text, model = service.process_audio_input(
                audio_bytes=b"dummy_wav_bytes",
                model_name="gemini-2.5-flash-preview-tts",
            )
            # The service MUST intercept and call execute with AUDIO_INPUT_MODEL
            called_model = mock_exec.call_args[1]["model_name"]
            assert called_model != "gemini-2.5-flash-preview-tts"
            assert "-tts" not in called_model.lower()
            assert called_model == config.AUDIO_INPUT_MODEL

    def test_discovery_excludes_tts_from_audio_input_models(self):
        """Model discovery must classify -tts models strictly as TTS and not as audio input."""
        from app.utils.gemini_discovery import _parse_model_object

        tts_data = {
            "name": "models/gemini-2.5-flash-preview-tts",
            "displayName": "Gemini 2.5 Flash Preview TTS",
            "supportedGenerationMethods": ["generateContent", "generateSpeech"],
        }
        model = _parse_model_object(tts_data)
        assert model.supports_tts is True
        assert model.supports_audio_input is False
        assert model.supports_text is False  # Must not be used for general chat


# ---------------------------------------------------------------------------
# TEST 3: Audio Output (TTS Service)
# ---------------------------------------------------------------------------
class TestTTSOutput:
    def test_clean_text_for_speech_strips_markdown_and_code(self):
        raw = "Here is some code:\n```python\nprint('hello')\n```\nVisit https://example.com for **details**!"
        cleaned = TTSService.clean_text_for_speech(raw)
        assert "print('hello')" not in cleaned
        assert "code block omitted" in cleaned
        assert "https://example.com" not in cleaned
        assert "**details**" not in cleaned
        assert "details" in cleaned

    def test_generate_browser_tts_js(self):
        js = TTSService.generate_browser_tts_js("Take a gentle deep breath.")
        assert "speechSynthesis" in js
        assert "SpeechSynthesisUtterance" in js


# ---------------------------------------------------------------------------
# TEST 4: Image Understanding
# ---------------------------------------------------------------------------
class TestImageUnderstanding:
    def test_gemini_image_analysis_uses_image_model(self):
        service = GeminiService(api_key="valid-key")
        with patch.object(service, "_execute_generate_content") as mock_exec:
            mock_exec.return_value = ("A magnificent African lion in the savanna.", config.IMAGE_MODEL)

            # Create tiny test image bytes
            img = Image.new("RGB", (10, 10), color="orange")
            buf = io.BytesIO()
            img.save(buf, format="PNG")

            text, model = service.analyze_image(
                image_bytes=buf.getvalue(),
                user_prompt="Describe this animal",
            )
            assert "African lion" in text
            assert model == config.IMAGE_MODEL


# ---------------------------------------------------------------------------
# TEST 5: Gemini Quota Exceeded & Automatic Local Fallback
# ---------------------------------------------------------------------------
class TestGeminiQuotaAndFallback:
    def test_detects_daily_quota_and_locks_manager(self):
        """Simulate a 429 daily quota error, verifying quota_manager locks and stops retries."""
        qm = QuotaManager(default_cooldown_seconds=3600)
        assert qm.is_daily_quota_active() is False

        # Simulate provider-supplied 17 hours retry delay
        qm.mark_daily_quota(retry_after_seconds=17 * 3600, error_message="Daily free-tier request limit exceeded.")
        assert qm.is_daily_quota_active() is True
        assert qm.is_gemini_available() is False
        assert "~17h" in qm.get_reset_display()

    @patch("services.gemini_service.GeminiService.generate_text")
    @patch("services.ollama_service.OllamaService.generate_text")
    @patch("services.ollama_service.OllamaService.is_running", return_value=True)
    def test_automatic_switch_to_ollama_on_daily_quota(
        self, mock_running, mock_ollama_gen, mock_gemini_gen
    ):
        """When Gemini throws daily quota, AIRouter automatically invokes Ollama."""
        mock_gemini_gen.side_effect = GeminiDailyQuotaExhaustedError("Daily quota exceeded", retry_after=3600)
        mock_ollama_gen.return_value = ("Response generated locally by Ollama Gemma 2B.", "gemma2:2b")

        router = AIRouter()
        result = router.route_turn(prompt="What is box breathing?", messages_history=[])

        assert result.success is True
        assert result.provider == "ollama"
        assert result.is_fallback is True
        assert result.model_used == "gemma2:2b"
        assert "locally by Ollama" in result.response_text


# ---------------------------------------------------------------------------
# TEST 6: Ollama Unavailable
# ---------------------------------------------------------------------------
class TestOllamaUnavailable:
    @patch("services.ollama_service.OllamaService.is_running", return_value=False)
    def test_helpful_notification_when_ollama_server_offline(self, mock_running):
        """When Gemini is exhausted and Ollama is offline, return helpful instructions without crashing."""
        quota_manager.mark_daily_quota(retry_after_seconds=3600)

        router = AIRouter()
        result = router.route_turn(prompt="Help me with anxiety", messages_history=[])

        assert result.success is False
        assert result.error_type == "ollama_offline"
        assert "ollama serve" in result.response_text


# ---------------------------------------------------------------------------
# TEST 7: Both Providers Unavailable
# ---------------------------------------------------------------------------
class TestBothProvidersUnavailable:
    @patch("config.GEMINI_API_KEY", "")
    @patch("services.ollama_service.OllamaService.is_running", return_value=False)
    def test_graceful_handling_without_crash(self, mock_ollama_running):
        router = AIRouter()
        result = router.route_turn(prompt="Hello", messages_history=[])
        assert result.success is False
        assert "No AI Provider Available" in result.response_text


# ---------------------------------------------------------------------------
# TEST 8: Provider Recovery
# ---------------------------------------------------------------------------
class TestProviderRecovery:
    def test_auto_recovery_when_cooldown_expires(self):
        qm = QuotaManager(default_cooldown_seconds=0.01)
        qm.mark_daily_quota(retry_after_seconds=0.01)
        assert qm.is_daily_quota_active() is True

        time.sleep(0.03)  # Wait for cooldown to expire
        # is_daily_quota_active should auto-recover!
        assert qm.is_daily_quota_active() is False
        assert qm.get_reset_display() == "Quota available"


# ---------------------------------------------------------------------------
# TEST 9: Multimodal Capability Enforcement
# ---------------------------------------------------------------------------
class TestMultimodalCapabilityEnforcement:
    def test_text_only_ollama_model_rejects_vision_with_clear_error(self):
        """Gemma 2B must not falsely claim image support."""
        ollama = OllamaService(text_model="gemma2:2b", vision_model="gemma2:2b")
        with patch.object(ollama, "is_running", return_value=True):
            with pytest.raises(OllamaUnsupportedModalityError) as exc_info:
                ollama.analyze_image(b"fake_image", model_name="gemma2:2b")
            assert "text-only model and does not support image analysis" in str(exc_info.value)
            assert "moondream" in str(exc_info.value)
