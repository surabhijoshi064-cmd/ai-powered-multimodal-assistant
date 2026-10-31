"""
Tests for Gemini rate-limiting, retry logic, and model fallback behaviour.
All tests use mocks — no real Gemini API calls are made.
"""
import time
import pytest
from collections import deque
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# GeminiRateLimiter unit tests
# ---------------------------------------------------------------------------

class TestGeminiRateLimiter:
    """Unit tests for the token-bucket-style rolling-window rate limiter."""

    def _make_limiter(self, rpm: int = 3):
        from app.utils.rate_limiter import GeminiRateLimiter
        return GeminiRateLimiter(requests_per_minute=rpm, window_seconds=60)

    def test_allows_requests_under_limit(self):
        limiter = self._make_limiter(rpm=3)
        for _ in range(3):
            limiter.wait_if_needed()
            limiter.record_request()
        assert limiter.current_count() == 3

    def test_current_count_matches_recorded(self):
        limiter = self._make_limiter(rpm=5)
        for _ in range(4):
            limiter.record_request()
        assert limiter.current_count() == 4

    def test_seconds_until_next_slot_is_zero_when_under_limit(self):
        limiter = self._make_limiter(rpm=5)
        limiter.record_request()
        assert limiter.seconds_until_next_slot() == 0.0

    def test_seconds_until_next_slot_positive_when_at_limit(self):
        from app.utils.rate_limiter import GeminiRateLimiter
        limiter = GeminiRateLimiter(requests_per_minute=2, window_seconds=60)
        now = time.monotonic()
        limiter._timestamps.append(now - 5)
        limiter._timestamps.append(now - 1)
        secs = limiter.seconds_until_next_slot()
        assert secs > 0, f"Expected >0 seconds, got {secs}"

    def test_purge_removes_expired_timestamps(self):
        from app.utils.rate_limiter import GeminiRateLimiter
        limiter = GeminiRateLimiter(requests_per_minute=2, window_seconds=60)
        limiter._timestamps.append(time.monotonic() - 61)
        assert limiter.current_count() == 0, "Expired timestamp should be purged"


# ---------------------------------------------------------------------------
# call_gemini_with_retry integration tests (fully mocked)
# ---------------------------------------------------------------------------

class TestCallGeminiWithRetry:
    """Tests for the centralized retry/fallback wrapper."""

    def _resource_exhausted_exc(self, msg: str = "429 RESOURCE_EXHAUSTED quota exceeded"):
        return Exception(msg)

    def _daily_quota_exc(self):
        return Exception(
            "429 RESOURCE_EXHAUSTED: generate_content_free_tier_requests "
            "Quota exceeded for quota metric. Retry in 17h 0m 0s"
        )

    def _not_found_exc(self, model: str = "gemini-fake"):
        return Exception(f"404 not found: models/{model}")

    def _auth_exc(self):
        return Exception("401 UNAUTHENTICATED: Invalid API key")

    @pytest.fixture(autouse=True)
    def patch_rate_limiter(self):
        with patch("app.utils.gemini_client.gemini_rate_limiter") as mock_rl:
            mock_rl.wait_if_needed = MagicMock()
            mock_rl.record_request = MagicMock()
            yield mock_rl

    def _make_genai(self, side_effects):
        """Build a fake genai module with given send_message side effects."""
        response_mock = MagicMock()
        response_mock.text = "Hello from Gemini!"
        if not isinstance(side_effects, (list, tuple)):
            side_effects = [side_effects] * 20  # repeat for all model attempts
        chat_mock = MagicMock()
        chat_mock.send_message.side_effect = side_effects
        genai_mock = MagicMock()
        genai_mock.GenerativeModel.return_value.start_chat.return_value = chat_mock
        return genai_mock

    def test_success_on_first_attempt(self):
        from app.utils.gemini_client import call_gemini_with_retry

        response_mock = MagicMock()
        response_mock.text = "  Gemini reply  "
        genai_mock = self._make_genai([response_mock])

        text, model_used = call_gemini_with_retry(
            genai_module=genai_mock,
            primary_model="gemini-1.5-flash",
            parts=["Hello"],
            max_retries=2,
        )
        assert text == "Gemini reply"
        assert "gemini-1.5-flash" in model_used

    def test_retries_on_per_minute_429_then_succeeds(self):
        """Should retry on RPM 429 and return success on second attempt."""
        from app.utils.gemini_client import call_gemini_with_retry

        response_mock = MagicMock()
        response_mock.text = "OK after retry"

        side_effects = [self._resource_exhausted_exc(), response_mock]
        genai_mock = self._make_genai(side_effects)

        with patch("app.utils.gemini_client.time.sleep"):
            text, _ = call_gemini_with_retry(
                genai_module=genai_mock,
                primary_model="gemini-1.5-flash",
                parts=["test"],
                max_retries=3,
            )
        assert text == "OK after retry"

    def test_daily_quota_raises_immediately_no_retry(self):
        """Daily quota error must raise GeminiRateLimitError immediately — no sleep."""
        from app.utils.gemini_client import call_gemini_with_retry, GeminiRateLimitError

        genai_mock = self._make_genai(self._daily_quota_exc())
        sleep_mock = MagicMock()

        with patch("app.utils.gemini_client.time.sleep", sleep_mock):
            with pytest.raises(GeminiRateLimitError) as exc_info:
                call_gemini_with_retry(
                    genai_module=genai_mock,
                    primary_model="gemini-1.5-flash",
                    parts=["test"],
                    max_retries=3,
                )

        exc = exc_info.value
        assert exc.is_daily_quota is True, "is_daily_quota must be True for daily limit"
        assert exc.retry_after > 3600, "retry_after must exceed 1 hour for daily quota"
        sleep_mock.assert_not_called()  # must NOT sleep and retry

    def test_raises_rate_limit_error_after_max_retries(self):
        """Should raise GeminiRateLimitError when RPM retries exhausted."""
        from app.utils.gemini_client import call_gemini_with_retry, GeminiRateLimitError

        genai_mock = self._make_genai(self._resource_exhausted_exc())

        with patch("app.utils.gemini_client.time.sleep"):
            with pytest.raises(GeminiRateLimitError) as exc_info:
                call_gemini_with_retry(
                    genai_module=genai_mock,
                    primary_model="gemini-1.5-flash",
                    parts=["test"],
                    max_retries=1,
                )
        assert exc_info.value.is_daily_quota is False

    def test_404_does_not_sleep_retry(self):
        """A 404 error must skip the model immediately without sleeping."""
        from app.utils.gemini_client import call_gemini_with_retry
        from app.utils.gemini_client import GeminiAllModelsExhaustedError, GeminiRateLimitError

        genai_mock = self._make_genai(self._not_found_exc())
        sleep_mock = MagicMock()

        with patch("app.utils.gemini_client.time.sleep", sleep_mock):
            with pytest.raises((GeminiAllModelsExhaustedError, GeminiRateLimitError, RuntimeError)):
                call_gemini_with_retry(
                    genai_module=genai_mock,
                    primary_model="gemini-fake-only",
                    parts=["test"],
                    max_retries=2,
                )
        sleep_mock.assert_not_called()

    def test_auth_error_raises_immediately_no_retry(self):
        """Auth errors (401) must raise GeminiAuthError immediately without retrying."""
        from app.utils.gemini_client import call_gemini_with_retry, GeminiAuthError

        genai_mock = self._make_genai(self._auth_exc())
        sleep_mock = MagicMock()

        with patch("app.utils.gemini_client.time.sleep", sleep_mock):
            with pytest.raises(GeminiAuthError):
                call_gemini_with_retry(
                    genai_module=genai_mock,
                    primary_model="gemini-1.5-flash",
                    parts=["test"],
                    max_retries=3,
                )
        sleep_mock.assert_not_called()
