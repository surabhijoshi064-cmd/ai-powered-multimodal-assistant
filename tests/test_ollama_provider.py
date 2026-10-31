"""
Tests for Ollama provider integration.

All tests use mocks — no live Ollama server is required.
Covers:
  - OllamaLLMProvider text generation (success + errors)
  - Gemini → Ollama daily-quota fallback in ChatService
  - Provider state tracking
  - Capability flags (vision, image gen)
  - Error type classification
"""
import os
import time
import pytest
from unittest.mock import MagicMock, patch

os.environ["TESTING"] = "true"


# ---------------------------------------------------------------------------
# Ollama client unit tests
# ---------------------------------------------------------------------------


class TestOllamaClientHealth:
    """Test health-check and model-listing utilities."""

    @patch("app.utils.ollama_client.httpx.get")
    def test_check_ollama_health_returns_true_on_200(self, mock_get):
        from app.utils.ollama_client import check_ollama_health
        mock_get.return_value = MagicMock(status_code=200)
        assert check_ollama_health() is True

    @patch("app.utils.ollama_client.httpx.get", side_effect=Exception("refused"))
    def test_check_ollama_health_returns_false_on_error(self, mock_get):
        from app.utils.ollama_client import check_ollama_health
        assert check_ollama_health() is False

    @patch("app.utils.ollama_client.httpx.get")
    def test_list_ollama_models_returns_names(self, mock_get):
        from app.utils.ollama_client import list_ollama_models
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"models": [{"name": "gemma2:2b"}, {"name": "llava:7b"}]},
        )
        mock_get.return_value.raise_for_status = lambda: None
        models = list_ollama_models()
        assert "gemma2:2b" in models
        assert "llava:7b" in models

    @patch("app.utils.ollama_client.httpx.get", side_effect=Exception("down"))
    def test_list_ollama_models_returns_empty_on_error(self, _):
        from app.utils.ollama_client import list_ollama_models
        assert list_ollama_models() == []


class TestOllamaVisionCapability:
    """model_supports_vision should only return True for vision model prefixes."""

    def test_llava_supports_vision(self):
        from app.utils.ollama_client import model_supports_vision
        assert model_supports_vision("llava:7b") is True

    def test_moondream_supports_vision(self):
        from app.utils.ollama_client import model_supports_vision
        assert model_supports_vision("moondream") is True

    def test_gemma_does_not_support_vision(self):
        from app.utils.ollama_client import model_supports_vision
        assert model_supports_vision("gemma2:2b") is False

    def test_llama_does_not_support_vision(self):
        from app.utils.ollama_client import model_supports_vision
        assert model_supports_vision("llama3.2:3b") is False


class TestChatWithOllama:
    """Test the OpenAI-compatible chat endpoint."""

    @patch("app.utils.ollama_client.httpx.Client")
    def test_successful_chat_returns_text_and_model(self, mock_client_cls):
        from app.utils.ollama_client import chat_with_ollama

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from Ollama!"}}],
            "model": "gemma2:2b",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        text, model = chat_with_ollama(
            messages=[{"role": "user", "content": "hi"}],
            model="gemma2:2b",
        )
        assert text == "Hello from Ollama!"
        assert model == "gemma2:2b"

    @patch("app.utils.ollama_client.httpx.Client")
    def test_404_raises_model_not_found_error(self, mock_client_cls):
        import httpx
        from app.utils.ollama_client import chat_with_ollama, OllamaModelNotFoundError

        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        with pytest.raises(OllamaModelNotFoundError):
            chat_with_ollama(
                messages=[{"role": "user", "content": "hi"}],
                model="notpulled:latest",
            )

    @patch("app.utils.ollama_client.httpx.Client")
    def test_connect_error_raises_unavailable(self, mock_client_cls):
        import httpx
        from app.utils.ollama_client import chat_with_ollama, OllamaUnavailableError

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client_cls.return_value = mock_client

        with pytest.raises(OllamaUnavailableError):
            chat_with_ollama(
                messages=[{"role": "user", "content": "hi"}],
                model="gemma2:2b",
            )

    @patch("app.utils.ollama_client.httpx.Client")
    def test_timeout_raises_timeout_error(self, mock_client_cls):
        import httpx
        from app.utils.ollama_client import chat_with_ollama, OllamaTimeoutError

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ReadTimeout("timed out")
        mock_client_cls.return_value = mock_client

        with pytest.raises(OllamaTimeoutError):
            chat_with_ollama(
                messages=[{"role": "user", "content": "hi"}],
                model="gemma2:2b",
            )


# ---------------------------------------------------------------------------
# Provider state tests
# ---------------------------------------------------------------------------


class TestProviderState:
    """Test in-process provider state tracking."""

    def setup_method(self):
        from app.utils.provider_state import clear_gemini_daily_quota_flag
        clear_gemini_daily_quota_flag()

    def test_initially_quota_not_exhausted(self):
        from app.utils.provider_state import is_gemini_daily_quota_exhausted
        assert is_gemini_daily_quota_exhausted() is False

    def test_mark_quota_sets_flag(self):
        from app.utils.provider_state import (
            mark_gemini_daily_quota_exhausted,
            is_gemini_daily_quota_exhausted,
        )
        mark_gemini_daily_quota_exhausted(retry_after_seconds=3600)
        assert is_gemini_daily_quota_exhausted() is True

    def test_clear_quota_removes_flag(self):
        from app.utils.provider_state import (
            mark_gemini_daily_quota_exhausted,
            clear_gemini_daily_quota_flag,
            is_gemini_daily_quota_exhausted,
        )
        mark_gemini_daily_quota_exhausted()
        clear_gemini_daily_quota_flag()
        assert is_gemini_daily_quota_exhausted() is False

    def test_quota_auto_clears_after_window(self):
        """Flag should auto-clear when the estimated reset time passes."""
        from app.utils.provider_state import (
            mark_gemini_daily_quota_exhausted,
            is_gemini_daily_quota_exhausted,
        )
        mark_gemini_daily_quota_exhausted(retry_after_seconds=0)  # immediate expiry
        time.sleep(0.05)
        assert is_gemini_daily_quota_exhausted() is False

    def test_set_and_get_active_provider(self):
        from app.utils.provider_state import set_active_provider, get_active_provider
        set_active_provider("ollama")
        assert get_active_provider() == "ollama"

    def test_get_provider_status_snapshot(self):
        from app.utils.provider_state import (
            mark_gemini_daily_quota_exhausted,
            get_provider_status_snapshot,
            set_active_provider,
        )
        mark_gemini_daily_quota_exhausted(retry_after_seconds=7200)
        set_active_provider("ollama")
        snap = get_provider_status_snapshot()
        assert snap["gemini_daily_quota_exhausted"] is True
        assert snap["active_provider"] == "ollama"
        assert snap["gemini_quota_reset_eta_hours"] > 0


# ---------------------------------------------------------------------------
# OllamaLLMProvider unit tests
# ---------------------------------------------------------------------------


class TestOllamaLLMProvider:
    """Test the provider adapter layer."""

    @patch("app.utils.ollama_client.httpx.Client")
    def test_generate_response_success(self, mock_client_cls):
        from app.providers.llm_provider import OllamaLLMProvider

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "I am a local AI."}}],
            "model": "gemma2:2b",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        provider = OllamaLLMProvider(model="gemma2:2b")
        result = provider.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert "local AI" in result

    @patch("app.utils.ollama_client.httpx.Client")
    def test_generate_response_unavailable_raises_runtime_error(self, mock_client_cls):
        import httpx
        from app.providers.llm_provider import OllamaLLMProvider

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("refused")
        mock_client_cls.return_value = mock_client

        provider = OllamaLLMProvider(model="gemma2:2b")
        with pytest.raises(RuntimeError) as exc_info:
            provider.generate_response(
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert "ollama_unavailable" in str(exc_info.value)

    @patch("app.utils.ollama_client.httpx.Client")
    def test_generate_response_model_not_found_raises_runtime_error(self, mock_client_cls):
        from app.providers.llm_provider import OllamaLLMProvider

        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        provider = OllamaLLMProvider(model="notpulled:latest")
        with pytest.raises(RuntimeError) as exc_info:
            provider.generate_response(
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert "ollama_model_not_found" in str(exc_info.value)

    def test_vision_capability_reflects_model_name(self):
        from app.providers.llm_provider import OllamaLLMProvider
        text_only = OllamaLLMProvider(model="gemma2:2b")
        assert text_only.supports_vision is False

        vision_model = OllamaLLMProvider(model="llava:7b")
        assert vision_model.supports_vision is True


# ---------------------------------------------------------------------------
# get_llm_provider() routing tests
# ---------------------------------------------------------------------------


class TestGetLLMProviderRouting:
    """Test that get_llm_provider() correctly routes based on settings and quota state."""

    def setup_method(self):
        from app.utils.provider_state import clear_gemini_daily_quota_flag
        clear_gemini_daily_quota_flag()
        # Ensure test mode is active
        os.environ["TESTING"] = "true"

    def test_returns_mock_in_test_mode(self):
        from app.providers.llm_provider import get_llm_provider, MockLLMProvider
        provider = get_llm_provider()
        assert isinstance(provider, MockLLMProvider)

    @patch.dict(os.environ, {"TESTING": "", "PYTEST_CURRENT_TEST": ""})
    @patch("app.providers.llm_provider.settings")
    def test_routes_to_ollama_when_daily_quota_exhausted(self, mock_settings):
        from app.providers.llm_provider import get_llm_provider, OllamaLLMProvider
        from app.utils.provider_state import mark_gemini_daily_quota_exhausted

        mock_settings.GEMINI_API_KEY = "key"
        mock_settings.LLM_PROVIDER = "mock"
        mock_settings.OLLAMA_ENABLED = True
        mock_settings.OLLAMA_FALLBACK_ON_DAILY_QUOTA = True
        mock_settings.OLLAMA_MODEL = "gemma2:2b"
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_TIMEOUT = 120.0
        mock_settings.LOG_LEVEL = "INFO"

        mark_gemini_daily_quota_exhausted(retry_after_seconds=3600)
        provider = get_llm_provider()
        assert isinstance(provider, OllamaLLMProvider)

    @patch.dict(os.environ, {"TESTING": "", "PYTEST_CURRENT_TEST": ""})
    @patch("app.providers.llm_provider.settings")
    def test_routes_to_gemini_when_quota_not_exhausted(self, mock_settings):
        from app.providers.llm_provider import get_llm_provider, GeminiLLMProvider

        mock_settings.GEMINI_API_KEY = "AIzaFake"
        mock_settings.LLM_PROVIDER = "gemini"
        mock_settings.OLLAMA_ENABLED = False
        mock_settings.OLLAMA_FALLBACK_ON_DAILY_QUOTA = False
        mock_settings.GEMINI_MODEL = "gemini-1.5-flash"
        mock_settings.LOG_LEVEL = "INFO"

        with patch("app.providers.llm_provider.GeminiLLMProvider") as mock_gemini_cls:
            mock_gemini_cls.return_value = MagicMock(spec=GeminiLLMProvider)
            provider = get_llm_provider()
            mock_gemini_cls.assert_called_once()

