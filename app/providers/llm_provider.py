from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import httpx
from app.config import settings
from app.utils.logging_config import setup_logging
from app.utils.gemini_client import (
    call_gemini_with_retry,
    GeminiRateLimitError,
    GeminiAuthError,
    GeminiPermissionError,
    GeminiNetworkError,
    GeminiModelNotFoundError,
    GeminiServerError,
)
from app.utils.ollama_client import (
    chat_with_ollama,
    chat_with_ollama_native,
    OllamaUnavailableError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaResponseError,
    list_ollama_models,
    model_supports_vision,
)
from app.utils.provider_state import (
    mark_gemini_daily_quota_exhausted,
    is_gemini_daily_quota_exhausted,
    get_quota_reset_eta_hours,
    set_active_provider,
)

logger = setup_logging(settings.LOG_LEVEL)


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Generate response given a list of conversation messages."""
        pass


class GeminiLLMProvider(BaseLLMProvider):
    """
    Adapter for Google Gemini API using the official google-generativeai SDK.
    Converts OpenAI-style message history into Gemini's multi-turn chat format.
    Set GEMINI_API_KEY and optionally GEMINI_MODEL in .env to activate.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        try:
            import google.generativeai as genai
            self._genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai is not installed. "
                "Run: pip install google-generativeai"
            )

        self.api_key = api_key or settings.GEMINI_API_KEY
        raw_model = model or settings.GEMINI_MODEL
        if raw_model in ("gemini-1.5-pro", "models/gemini-1.5-pro"):
            raw_model = "gemini-1.5-pro-001"
        self.model_name = raw_model

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Add it to your .env file to use the Gemini provider."
            )

        self._genai.configure(api_key=self.api_key)
        logger.info(f"GeminiLLMProvider initialized with model '{self.model_name}'.")

    def _to_gemini_history(
        self, messages: List[Dict[str, str]]
    ) -> tuple[list, str]:
        """
        Splits OpenAI-style messages into (history_for_chat, last_user_text).
        Gemini uses role='user'|'model' (not 'assistant').
        The final user turn is sent via chat.send_message() separately.
        """
        gemini_history = []
        for msg in messages[:-1]:  # all turns except the last one
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [{"text": msg["content"]}],
            })
        last_user_text = messages[-1]["content"] if messages else ""
        return gemini_history, last_user_text

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        if not messages:
            return "I did not receive any message to respond to."

        gemini_history, last_user_text = self._to_gemini_history(messages)

        generation_config = self._genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        try:
            text, used_model = call_gemini_with_retry(
                genai_module=self._genai,
                primary_model=self.model_name,
                parts=[last_user_text],
                history=gemini_history,
                generation_config=generation_config,
                system_instruction=system_prompt or "",
                api_key=self.api_key,
            )
            if used_model != self.model_name:
                logger.info(f"Fallback: used '{used_model}' instead of '{self.model_name}'.")
            return text

        except GeminiRateLimitError as exc:
            logger.warning(f"GeminiLLMProvider rate-limited: {exc}")
            if exc.is_daily_quota:
                raise RuntimeError(
                    f"daily_quota::{int(exc.retry_after)}::"
                    "Gemini's daily free-tier quota has been reached. "
                    "Google restricts further requests until the quota resets."
                ) from exc
            raise RuntimeError(
                f"rate_limit::{int(exc.retry_after)}::"
                "Gemini per-minute quota reached. Please wait and retry."
            ) from exc

        except GeminiAuthError as exc:
            logger.error(f"GeminiLLMProvider auth error: {exc}")
            raise RuntimeError(
                f"auth_failure::{exc}::"
                "Gemini authentication failed. Please verify your GEMINI_API_KEY."
            ) from exc

        except GeminiPermissionError as exc:
            logger.error(f"GeminiLLMProvider permission error: {exc}")
            raise RuntimeError(
                f"permission_denied::{exc}::"
                "Access to Google Gemini API is denied (HTTP 403). Check project permissions."
            ) from exc

        except GeminiNetworkError as exc:
            logger.error(f"GeminiLLMProvider network error: {exc}")
            raise RuntimeError(
                f"network_error::{exc}::"
                "Network connection to Google Gemini API failed."
            ) from exc

        except GeminiServerError as exc:
            logger.error(f"GeminiLLMProvider server error: {exc}")
            raise RuntimeError(
                "Google's Gemini service is temporarily unavailable. Please try again in a few minutes."
            ) from exc

        except GeminiModelNotFoundError as exc:
            logger.error(f"GeminiLLMProvider — models not found: {exc}")
            raise RuntimeError(
                f"model_not_found::{exc}::"
                "No candidate Gemini model was reachable on this account."
            ) from exc


class OpenAILLMProvider(BaseLLMProvider):
    """
    Adapter for OpenAI and OpenAI-compatible endpoints (Groq, Ollama, vLLM, etc.).
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or settings.LLM_API_KEY
        self.model = model or settings.LLM_MODEL
        self.base_url = (base_url or settings.LLM_BASE_URL).rstrip("/") or "https://api.openai.com/v1"

        if not self.api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            logger.warning("OpenAI LLM provider initialized without API key.")

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        if not self.api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            raise ValueError(
                "LLM API key is not configured. Please set LLM_API_KEY in your environment or .env file."
            )

        payload_messages = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})
        payload_messages.extend(messages)

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
        }
        data = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API HTTP error: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Language model service error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"LLM API connection error: {str(e)}")
            raise RuntimeError(
                "Failed to connect to the language model provider. "
                "Please check network and endpoint configuration."
            )


class MockLLMProvider(BaseLLMProvider):
    """
    Explicit development test double for local automated testing and environments
    without external API keys. Returns [Test Double] prefixed responses.
    Only used when LLM_PROVIDER=mock and no GEMINI_API_KEY is present.
    """
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        last_message = messages[-1]["content"] if messages else ""
        lower_msg = last_message.lower()

        if "breathing" in lower_msg or "calm" in lower_msg or "anxious" in lower_msg:
            return (
                "[Test Double]: I hear you, and it is completely understandable to feel overwhelmed. "
                "Let's take a slow breath together: Inhale for 4 seconds, hold for 4 seconds, "
                "and exhale gently for 4 seconds. How are you feeling right now?"
            )
        elif "hello" in lower_msg or "hi" in lower_msg:
            return "[Test Double]: Hello! I am here to support you and help answer any questions or discuss your documents."
        else:
            return f"[Test Double]: Thank you for sharing: \"{last_message}\". I am listening and here to support you."


class OllamaLLMProvider(BaseLLMProvider):
    """
    Adapter for Ollama (locally-running open-source LLMs).

    Used as automatic fallback when Gemini hits its daily quota limit,
    or as a primary provider when OLLAMA_ENABLED=true and LLM_PROVIDER=ollama.

    Capabilities
    ------------
    - Text generation: always supported.
    - Vision (image understanding): only for llava/bakllava/moondream models.
    - Image generation: NOT supported (Ollama is not a diffusion server).
    - Audio transcription: NOT natively supported via this provider.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.model = model or settings.OLLAMA_MODEL
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.OLLAMA_TIMEOUT
        self.supports_vision = model_supports_vision(self.model)

        logger.info(
            f"OllamaLLMProvider initialized: model='{self.model}', "
            f"base_url='{self.base_url}', vision={self.supports_vision}"
        )

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        if not messages:
            return "I did not receive any message to respond to."

        # Try OpenAI-compatible endpoint first, then fall back to native endpoint
        try:
            text, used_model = chat_with_ollama(
                messages=messages,
                model=self.model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            if used_model != self.model:
                logger.info(f"Ollama used model '{used_model}' (requested '{self.model}').")
            set_active_provider("ollama")
            return text

        except OllamaResponseError:
            # /v1 endpoint may not exist on older Ollama — retry with native API
            logger.debug("OpenAI-compat endpoint failed; retrying with native /api/chat.")
            try:
                text, _ = chat_with_ollama_native(
                    messages=messages,
                    model=self.model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
                set_active_provider("ollama")
                return text
            except OllamaUnavailableError as exc:
                raise RuntimeError(f"ollama_unavailable::{exc}") from exc
            except OllamaModelNotFoundError as exc:
                raise RuntimeError(f"ollama_model_not_found::{exc}") from exc
            except OllamaTimeoutError as exc:
                raise RuntimeError(f"ollama_timeout::{exc}") from exc
            except OllamaResponseError as exc:
                raise RuntimeError(f"ollama_error::{exc}") from exc

        except OllamaUnavailableError as exc:
            raise RuntimeError(f"ollama_unavailable::{exc}") from exc

        except OllamaModelNotFoundError as exc:
            raise RuntimeError(f"ollama_model_not_found::{exc}") from exc

        except OllamaTimeoutError as exc:
            raise RuntimeError(f"ollama_timeout::{exc}") from exc


def get_llm_provider() -> BaseLLMProvider:
    """
    Provider resolution order:
      0. If running in automated tests (pytest) → MockLLMProvider
      1. If Gemini daily quota is exhausted AND Ollama is enabled → OllamaLLMProvider
      2. If LLM_PROVIDER=gemini  → GeminiLLMProvider (requires GEMINI_API_KEY)
      3. If GEMINI_API_KEY set and LLM_PROVIDER=mock (default) → auto-upgrade to Gemini
      4. If LLM_PROVIDER=ollama  → OllamaLLMProvider
      5. If LLM_PROVIDER=openai|groq  → OpenAILLMProvider
      6. Fallback → MockLLMProvider (offline / testing)
    """
    import os
    if os.getenv("PYTEST_CURRENT_TEST") or os.getenv("TESTING") == "true":
        return MockLLMProvider()

    # --- Step 1: Gemini daily quota exhausted → use Ollama immediately (don't hammer Gemini) ---
    if (
        is_gemini_daily_quota_exhausted()
        and settings.OLLAMA_ENABLED
        and settings.OLLAMA_FALLBACK_ON_DAILY_QUOTA
    ):
        eta = get_quota_reset_eta_hours()
        logger.info(
            f"Gemini daily quota exhausted (resets in ~{eta}h). "
            "Automatically routing to OllamaLLMProvider."
        )
        return OllamaLLMProvider()

    provider = settings.LLM_PROVIDER.lower().strip()

    # Auto-upgrade: if the user set a Gemini key but left LLM_PROVIDER=mock (the default)
    if settings.GEMINI_API_KEY and provider == "mock":
        logger.info(
            "GEMINI_API_KEY detected — automatically switching provider to 'gemini'. "
            "Set LLM_PROVIDER=gemini in .env to suppress this message."
        )
        provider = "gemini"

    if provider == "gemini":
        if not settings.GEMINI_API_KEY:
            logger.warning(
                "LLM_PROVIDER=gemini but GEMINI_API_KEY is not configured. "
                "Falling back to MockLLMProvider."
            )
            return MockLLMProvider()
        try:
            return GeminiLLMProvider()
        except Exception as e:
            logger.error(f"Failed to initialize GeminiLLMProvider: {e}. Falling back to Mock.")
            return MockLLMProvider()

    elif provider == "ollama":
        return OllamaLLMProvider()

    elif provider in ["openai", "groq"]:
        return OpenAILLMProvider()

    elif provider == "mock":
        logger.info("LLM_PROVIDER=mock — using test double (no real AI calls).")
        return MockLLMProvider()

    else:
        logger.warning(f"Unknown LLM provider '{provider}', defaulting to MockLLMProvider.")
        return MockLLMProvider()


def get_ollama_provider_or_none() -> Optional[OllamaLLMProvider]:
    """
    Return an OllamaLLMProvider if OLLAMA_ENABLED=true, else None.
    Used by chat_service for explicit fallback routing.
    """
    if settings.OLLAMA_ENABLED:
        return OllamaLLMProvider()
    return None
