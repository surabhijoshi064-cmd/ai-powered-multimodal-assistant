"""
Comprehensive test suite for Gemini Model Discovery, Error Classification,
Text Generation, Image Understanding, and Image Generation.
All tests use deterministic mocks — no real API calls or quota consumption.
"""
import io
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from app.utils.gemini_discovery import (
    discover_models_rest,
    discover_models_sdk,
    discover_models,
    _parse_model_object,
    DiscoveredModel,
    ModelDiscoveryResult,
)
from app.utils.gemini_client import (
    call_gemini_with_retry,
    generate_image_with_gemini,
    build_candidate_list,
    GeminiAuthError,
    GeminiPermissionError,
    GeminiRateLimitError,
    GeminiNetworkError,
    GeminiModelNotFoundError,
    GeminiServerError,
    GeminiMissingKeyError,
    GeminiImageGenUnsupportedError,
)
from app.services.multimodal_service import MultimodalService


# ---------------------------------------------------------------------------
# 1. API Key Loading & Validation
# ---------------------------------------------------------------------------

class TestApiKeyHandling:
    def test_missing_api_key_returns_missing_key_error(self):
        result = discover_models(api_key="")
        assert result.success is False
        assert result.error_type == "missing_key"
        assert result.is_fallback is True

    def test_missing_api_key_generate_image_raises_error(self):
        with pytest.raises(GeminiMissingKeyError):
            generate_image_with_gemini(api_key="", prompt="test")


# ---------------------------------------------------------------------------
# 2. Model Metadata Parsing & Generation Methods
# ---------------------------------------------------------------------------

class TestModelMetadataParsing:
    def test_parses_text_generation_model(self):
        raw = {
            "name": "models/gemini-1.5-flash",
            "displayName": "Gemini 1.5 Flash",
            "description": "Fast multimodal model",
            "supportedGenerationMethods": ["generateContent", "countTokens"],
        }
        model = _parse_model_object(raw)
        assert model.clean_name == "gemini-1.5-flash"
        assert model.display_name == "Gemini 1.5 Flash"
        assert model.supports_text is True
        assert model.supports_vision is True
        assert model.supports_image_generation is False
        assert model.supports_embedding is False

    def test_parses_imagen_model(self):
        raw = {
            "name": "models/imagen-3.0-generate-002",
            "displayName": "Imagen 3",
            "description": "Image generation",
            "supportedGenerationMethods": ["imageGeneration"],
        }
        model = _parse_model_object(raw)
        assert model.clean_name == "imagen-3.0-generate-002"
        assert model.supports_text is False
        assert model.supports_image_generation is True

    def test_parses_embedding_model(self):
        raw = {
            "name": "models/text-embedding-004",
            "displayName": "Embedding 4",
            "description": "Embeddings",
            "supportedGenerationMethods": ["embedContent"],
        }
        model = _parse_model_object(raw)
        assert model.clean_name == "text-embedding-004"
        assert model.supports_text is False
        assert model.supports_embedding is True


# ---------------------------------------------------------------------------
# 3. Model Discovery REST & Pagination
# ---------------------------------------------------------------------------

class TestModelDiscoveryRest:
    @patch("app.utils.gemini_discovery.requests.get")
    def test_discovery_success_with_filtering(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {
                    "name": "models/gemini-1.5-flash",
                    "displayName": "Gemini 1.5 Flash",
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/imagen-3.0-generate-002",
                    "displayName": "Imagen 3",
                    "supportedGenerationMethods": ["imageGeneration"],
                },
                {
                    "name": "models/text-embedding-004",
                    "displayName": "Text Embedding",
                    "supportedGenerationMethods": ["embedContent"],
                },
            ]
        }
        mock_get.return_value = mock_resp

        result = discover_models_rest(api_key="valid-test-key")
        assert result.success is True
        assert "gemini-1.5-flash" in result.text_models
        assert "imagen-3.0-generate-002" in result.image_models
        assert "text-embedding-004" in result.embedding_models

    @patch("app.utils.gemini_discovery.requests.get")
    def test_discovery_pagination_handling(self, mock_get):
        page1 = MagicMock()
        page1.status_code = 200
        page1.json.return_value = {
            "models": [
                {"name": "models/gemini-1.5-flash", "supportedGenerationMethods": ["generateContent"]},
            ],
            "nextPageToken": "page-2-token",
        }

        page2 = MagicMock()
        page2.status_code = 200
        page2.json.return_value = {
            "models": [
                {"name": "models/gemini-1.5-pro", "supportedGenerationMethods": ["generateContent"]},
            ],
        }
        mock_get.side_effect = [page1, page2]

        result = discover_models_rest(api_key="valid-test-key")
        assert result.success is True
        assert len(result.text_models) == 2
        assert "gemini-1.5-flash" in result.text_models
        assert "gemini-1.5-pro" in result.text_models


# ---------------------------------------------------------------------------
# 4. Error Classification (Auth, 403, 404, 429, Network, Empty List)
# ---------------------------------------------------------------------------

class TestErrorClassification:
    @patch("app.utils.gemini_discovery.requests.get")
    def test_invalid_api_key_401(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "API key not valid. Please pass a valid API key."
        mock_get.return_value = mock_resp

        result = discover_models_rest(api_key="bad-key")
        assert result.success is False
        assert result.error_type == "auth_failure"
        assert "rejected" in result.error_message.lower() or "401" in result.error_message

    @patch("app.utils.gemini_discovery.requests.get")
    def test_permission_denied_403(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Generative Language API has not been used in project 123 before."
        mock_get.return_value = mock_resp

        result = discover_models_rest(api_key="no-perm-key")
        assert result.success is False
        assert result.error_type == "permission_denied"

    @patch("app.utils.gemini_discovery.requests.get")
    def test_quota_exceeded_429(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Resource exhausted: free_tier_requests daily limit exceeded."
        mock_get.return_value = mock_resp

        result = discover_models_rest(api_key="rate-key")
        assert result.success is False
        assert result.error_type == "quota_exceeded_daily"

    @patch("app.utils.gemini_discovery.requests.get")
    def test_network_failure(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Name resolution failed")

        result = discover_models_rest(api_key="net-key")
        assert result.success is False
        assert result.error_type == "network_error"
        assert result.is_fallback is True
        assert len(result.text_models) > 0  # provides fallback models

    @patch("app.utils.gemini_discovery.requests.get")
    def test_empty_model_list(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": []}
        mock_get.return_value = mock_resp

        result = discover_models_rest(api_key="empty-key")
        assert result.success is False
        assert result.error_type == "empty_list"


# ---------------------------------------------------------------------------
# 5. Text Generation & Model Fallback
# ---------------------------------------------------------------------------

class TestTextGenerationAndFallback:
    @patch("app.utils.gemini_client.gemini_rate_limiter")
    def test_text_generation_success(self, mock_rl):
        mock_rl.wait_if_needed = MagicMock()
        mock_rl.record_request = MagicMock()

        genai_mock = MagicMock()
        chat_mock = MagicMock()
        resp_mock = MagicMock()
        resp_mock.text = "Compassionate response from Gemini."
        chat_mock.send_message.return_value = resp_mock
        genai_mock.GenerativeModel.return_value.start_chat.return_value = chat_mock

        text, used_model = call_gemini_with_retry(
            genai_module=genai_mock,
            primary_model="gemini-1.5-flash",
            parts=["Hello"],
            max_retries=1,
        )
        assert text == "Compassionate response from Gemini."
        assert used_model == "gemini-1.5-flash"

    @patch("app.utils.gemini_client.gemini_rate_limiter")
    def test_model_not_found_advances_to_next_candidate(self, mock_rl):
        mock_rl.wait_if_needed = MagicMock()
        mock_rl.record_request = MagicMock()

        # First model returns 404, second model succeeds
        success_resp = MagicMock()
        success_resp.text = "Fallback success"

        chat_fail = MagicMock()
        chat_fail.send_message.side_effect = Exception("404 models/gemini-bad is not found")

        chat_success = MagicMock()
        chat_success.send_message.return_value = success_resp

        def model_side_effect(model_name, **kwargs):
            m = MagicMock()
            if "bad" in model_name:
                m.start_chat.return_value = chat_fail
            else:
                m.start_chat.return_value = chat_success
            return m

        genai_mock = MagicMock()
        genai_mock.GenerativeModel.side_effect = model_side_effect

        text, used_model = call_gemini_with_retry(
            genai_module=genai_mock,
            primary_model="gemini-bad",
            parts=["test"],
            max_retries=1,
        )
        assert text == "Fallback success"
        assert "bad" not in used_model

    @patch("app.utils.gemini_client.gemini_rate_limiter")
    def test_auth_error_not_retried(self, mock_rl):
        mock_rl.wait_if_needed = MagicMock()

        genai_mock = MagicMock()
        chat_mock = MagicMock()
        chat_mock.send_message.side_effect = Exception("401 API_KEY_INVALID: API key not valid")
        genai_mock.GenerativeModel.return_value.start_chat.return_value = chat_mock

        with pytest.raises(GeminiAuthError):
            call_gemini_with_retry(
                genai_module=genai_mock,
                primary_model="gemini-1.5-flash",
                parts=["test"],
                max_retries=3,
            )
        # Auth error fails on attempt 1, never retried 3 times
        assert chat_mock.send_message.call_count == 1


# ---------------------------------------------------------------------------
# 6. Image Understanding (Multimodal Vision)
# ---------------------------------------------------------------------------

class TestImageUnderstanding:
    @patch("app.services.multimodal_service.call_gemini_with_retry")
    @patch("app.services.multimodal_service.settings")
    def test_analyze_image_returns_text(self, mock_settings, mock_call):
        mock_settings.GEMINI_API_KEY = "dummy-key"
        mock_settings.GEMINI_MODEL = "gemini-1.5-flash"
        mock_call.return_value = ("A peaceful sunrise over a tranquil forest lake.", "gemini-1.5-flash")

        service = MultimodalService()
        # Create a tiny 10x10 test image
        img = Image.new("RGB", (10, 10), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = service.analyze_image(
            image_bytes=buf.getvalue(),
            filename="nature.png",
            user_prompt="Describe this scene",
        )
        assert "peaceful sunrise" in result


# ---------------------------------------------------------------------------
# 7. Image Generation (Text-to-Image with Imagen)
# ---------------------------------------------------------------------------

class TestImageGeneration:
    @patch("app.utils.gemini_client.requests.post")
    @patch("app.utils.gemini_client.gemini_rate_limiter")
    def test_image_generation_rest_success(self, mock_rl, mock_post):
        mock_rl.wait_if_needed = MagicMock()
        mock_rl.record_request = MagicMock()

        # Create a 5x5 test PNG and base64-encode it
        test_img = Image.new("RGB", (5, 5), color="green")
        buf = io.BytesIO()
        test_img.save(buf, format="PNG")
        import base64
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "predictions": [{"bytesBase64Encoded": b64_img}]
        }
        mock_post.return_value = mock_resp

        generated = generate_image_with_gemini(
            api_key="valid-key",
            prompt="A zen garden",
            model_name="imagen-3.0-generate-002",
        )
        assert isinstance(generated, Image.Image)
        assert generated.size == (5, 5)

    @patch("app.utils.gemini_client.requests.post")
    @patch("app.utils.gemini_client.gemini_rate_limiter")
    def test_image_generation_unsupported_account_raises_clean_error(self, mock_rl, mock_post):
        mock_rl.wait_if_needed = MagicMock()
        mock_rl.record_request = MagicMock()

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "models/imagen-3.0-generate-002 is not found for this account"
        mock_post.return_value = mock_resp

        with pytest.raises(GeminiImageGenUnsupportedError) as exc_info:
            generate_image_with_gemini(
                api_key="free-tier-key",
                prompt="A flying car",
                model_name="imagen-3.0-generate-002",
            )
        assert "not available" in str(exc_info.value).lower()
        # Verify it specifically mentions image understanding remains functional
        assert "image understanding" in str(exc_info.value).lower()

