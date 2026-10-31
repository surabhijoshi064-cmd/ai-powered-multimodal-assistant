"""
Ollama API client for MindfulAI.

Communicates with a locally-running Ollama server via its REST API
(compatible with OpenAI's /v1/chat/completions schema).

Only text generation is supported by default.
Vision (image understanding) is possible ONLY with specific multimodal models
(e.g. llava, bakllava, moondream). The client exposes capability flags so the
UI can warn users accurately — we never claim a feature works unless confirmed.

Reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OllamaError(RuntimeError):
    """Base exception for all Ollama failures."""


class OllamaUnavailableError(OllamaError):
    """Ollama server is not running or not reachable."""


class OllamaModelNotFoundError(OllamaError):
    """The requested model has not been pulled yet."""


class OllamaTimeoutError(OllamaError):
    """Request timed out (model is large or system is slow)."""


class OllamaResponseError(OllamaError):
    """Ollama returned an unexpected or malformed response."""


# ---------------------------------------------------------------------------
# Multimodal-capable model name prefixes
# Vision models support image understanding via base64-encoded images.
# ---------------------------------------------------------------------------
VISION_CAPABLE_MODEL_PREFIXES = (
    "llava",
    "bakllava",
    "moondream",
    "llava-phi",
    "llava-llama",
    "minicpm-v",
    "cogvlm",
    "internvl",
)


def model_supports_vision(model_name: str) -> bool:
    """Return True if the given Ollama model supports image understanding."""
    lower = model_name.lower()
    return any(lower.startswith(pfx) or pfx in lower for pfx in VISION_CAPABLE_MODEL_PREFIXES)


# ---------------------------------------------------------------------------
# Health & Model Discovery
# ---------------------------------------------------------------------------


def check_ollama_health(base_url: str = "http://localhost:11434", timeout: float = 3.0) -> bool:
    """
    Ping the Ollama server. Returns True if running, False otherwise.
    Never raises — designed for use in status indicators.
    """
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def list_ollama_models(base_url: str = "http://localhost:11434", timeout: float = 5.0) -> List[str]:
    """
    Return names of locally-pulled Ollama models.
    Returns an empty list if Ollama is unavailable.
    """
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as exc:
        logger.debug(f"Could not list Ollama models: {exc}")
        return []


def get_model_info(
    model_name: str,
    base_url: str = "http://localhost:11434",
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """
    Return the Ollama model info dict for a pulled model.
    Returns {} on failure.
    """
    try:
        r = httpx.post(
            f"{base_url.rstrip('/')}/api/show",
            json={"name": model_name},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.debug(f"Could not fetch model info for '{model_name}': {exc}")
        return {}


# ---------------------------------------------------------------------------
# Chat (OpenAI-compatible /v1/chat/completions endpoint)
# ---------------------------------------------------------------------------


def chat_with_ollama(
    messages: List[Dict[str, str]],
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> Tuple[str, str]:
    """
    Send a chat request to Ollama using the OpenAI-compatible endpoint.

    Parameters
    ----------
    messages      : List of {"role": "user"|"assistant", "content": "..."} dicts.
    model         : Model name as returned by `list_ollama_models()`.
    system_prompt : Optional system instruction prepended as a system message.
    temperature   : Sampling temperature (0–2).
    max_tokens    : Maximum tokens in the response.
    base_url      : Ollama server base URL (default: http://localhost:11434).
    timeout       : Request timeout in seconds; large models may need 120 s+.

    Returns
    -------
    (reply_text, model_name) tuple on success.

    Raises
    ------
    OllamaUnavailableError   : Server not reachable.
    OllamaModelNotFoundError : Model not pulled.
    OllamaTimeoutError       : Request timed out.
    OllamaResponseError      : Unexpected response format.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    payload_messages: List[Dict[str, str]] = []
    if system_prompt:
        payload_messages.append({"role": "system", "content": system_prompt})
    payload_messages.extend(messages)

    payload = {
        "model": model,
        "messages": payload_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)

        if response.status_code == 404:
            raise OllamaModelNotFoundError(
                f"Ollama model '{model}' is not available. "
                f"Pull it first: ollama pull {model}"
            )

        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise OllamaResponseError(f"Ollama returned an empty choices list: {data}")

        content = choices[0].get("message", {}).get("content", "").strip()
        if not content:
            raise OllamaResponseError("Ollama returned an empty response message.")

        used_model = data.get("model", model)
        logger.info(f"Ollama response received from model '{used_model}'.")
        return content, used_model

    except httpx.ConnectError as exc:
        raise OllamaUnavailableError(
            f"Cannot connect to Ollama at {base_url}. "
            "Make sure Ollama is running: run `ollama serve` in a terminal."
        ) from exc

    except httpx.TimeoutException as exc:
        raise OllamaTimeoutError(
            f"Ollama request timed out after {timeout}s for model '{model}'. "
            "The model may be loading for the first time or your system is under load. "
            "Try again in a moment."
        ) from exc

    except (OllamaModelNotFoundError, OllamaUnavailableError, OllamaResponseError):
        raise

    except httpx.HTTPStatusError as exc:
        raise OllamaResponseError(
            f"Ollama HTTP error {exc.response.status_code}: {exc.response.text[:200]}"
        ) from exc

    except Exception as exc:
        raise OllamaResponseError(f"Unexpected Ollama error: {exc}") from exc


# ---------------------------------------------------------------------------
# Native /api/chat endpoint (fallback for servers not supporting /v1)
# ---------------------------------------------------------------------------


def chat_with_ollama_native(
    messages: List[Dict[str, str]],
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> Tuple[str, str]:
    """
    Fallback using Ollama's native /api/chat endpoint (non-streaming).
    Used when the /v1/chat/completions endpoint is unavailable.
    """
    url = f"{base_url.rstrip('/')}/api/chat"

    native_messages = []
    if system_prompt:
        native_messages.append({"role": "system", "content": system_prompt})
    native_messages.extend(messages)

    payload = {
        "model": model,
        "messages": native_messages,
        "options": {"temperature": temperature},
        "stream": False,
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)

        if response.status_code == 404:
            raise OllamaModelNotFoundError(
                f"Ollama model '{model}' is not available. Pull it: ollama pull {model}"
            )

        response.raise_for_status()
        data = response.json()

        content = data.get("message", {}).get("content", "").strip()
        if not content:
            raise OllamaResponseError(f"Ollama native API returned empty content: {data}")

        used_model = data.get("model", model)
        return content, used_model

    except httpx.ConnectError as exc:
        raise OllamaUnavailableError(
            f"Cannot connect to Ollama at {base_url}. Run `ollama serve`."
        ) from exc

    except httpx.TimeoutException as exc:
        raise OllamaTimeoutError(
            f"Ollama native API timed out after {timeout}s for model '{model}'."
        ) from exc

    except (OllamaModelNotFoundError, OllamaUnavailableError, OllamaResponseError):
        raise

    except httpx.HTTPStatusError as exc:
        raise OllamaResponseError(
            f"Ollama native HTTP error {exc.response.status_code}: {exc.response.text[:200]}"
        ) from exc

    except Exception as exc:
        raise OllamaResponseError(f"Unexpected Ollama native error: {exc}") from exc

