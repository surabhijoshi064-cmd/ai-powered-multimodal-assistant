"""
Centralized Gemini API client wrapper with dynamic model discovery, retry, and image generation.

Both the FastAPI backend (GeminiLLMProvider) and Streamlit frontend use this module.
Dynamic model discovery replaces outdated hardcoded lists.

Error handling matrix:
  • 401 / Invalid Key        → GeminiAuthError (never retried)
  • 403 / Permission/Billing → GeminiPermissionError (never retried)
  • 429 / Daily Quota        → GeminiRateLimitError(is_daily_quota=True) (never retried)
  • 429 / RPM Limit          → exponential backoff + jitter, retried up to max_retries
  • 404 / Model Not Found    → try next discovered model; if all fail, GeminiModelNotFoundError
  • Network Connection       → GeminiNetworkError (never retried in a loop)
  • 500 / Server Error       → GeminiServerError
"""
import os
import io
import re
import time
import base64
import random
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from PIL import Image

from app.utils.rate_limiter import gemini_rate_limiter
from app.utils.gemini_discovery import (
    discover_models,
    get_available_text_models,
    get_available_image_models,
    DOCUMENTED_DEFAULT_TEXT_MODELS,
    DOCUMENTED_DEFAULT_IMAGE_MODELS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strict Exception Hierarchy
# ---------------------------------------------------------------------------


class GeminiError(RuntimeError):
    """Base exception for all Gemini API failures."""


class GeminiMissingKeyError(GeminiError):
    """Raised when GEMINI_API_KEY is not configured."""


class GeminiAuthError(GeminiError):
    """Raised on HTTP 401 or invalid API key."""


class GeminiPermissionError(GeminiError):
    """Raised on HTTP 403 permission denied or service disabled."""


class GeminiRateLimitError(GeminiError):
    """
    Raised when Gemini quota is exhausted.

    Attributes
    ----------
    retry_after : float
        Seconds to wait before retrying.
    is_daily_quota : bool
        True when the quota is daily (cannot be resolved by short waiting).
    """
    def __init__(self, message: str, retry_after: float = 60.0, is_daily_quota: bool = False):
        super().__init__(message)
        self.retry_after = retry_after
        self.is_daily_quota = is_daily_quota


class GeminiNetworkError(GeminiError):
    """Raised on network connectivity, DNS, or socket connection errors."""


class GeminiModelNotFoundError(GeminiError):
    """Raised when a specific model or all candidate models are not found (HTTP 404)."""


# Backwards compatibility alias
GeminiAllModelsExhaustedError = GeminiModelNotFoundError


class GeminiServerError(GeminiError):
    """Raised on HTTP 500/502/503/504 server errors."""


class GeminiEmptyModelListError(GeminiError):
    """Raised when the API key has no compatible models returned by Google."""


class GeminiImageGenUnsupportedError(GeminiError):
    """Raised when image generation is requested but Imagen is not enabled for this account."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
_BASE_BACKOFF = 15.0
_MAX_BACKOFF = 120.0
_DAILY_QUOTA_THRESHOLD_SECS = 600  # 10 minutes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_retry_after(exc: Exception) -> Optional[float]:
    """Extract retry delay (seconds) from structured exception details or message."""
    try:
        for detail in exc.details():  # type: ignore[attr-defined]
            if hasattr(detail, "retry_delay"):
                rd = detail.retry_delay
                secs = rd.seconds + rd.nanos / 1e9
                if secs > 0:
                    return secs
    except Exception:
        pass

    exc_str = str(exc)

    # Hours/Minutes pattern: "17h 0m"
    hours_match = re.search(r"(\d+)\s*h(?:our)?s?\s*(\d+)?\s*m", exc_str, re.I)
    if hours_match:
        hours = int(hours_match.group(1))
        minutes = int(hours_match.group(2) or 0)
        return float(hours * 3600 + minutes * 60)

    # Seconds pattern: "retry in 60s" or "retry after 60"
    match = re.search(r"retry[_\s-]*(?:after|in|delay)[:\s]+(\d+(?:\.\d+)?)\s*s?", exc_str, re.I)
    if match:
        return float(match.group(1))

    return None


def _is_daily_quota_message(exc: Exception) -> bool:
    """Detect whether this is a daily quota exhaustion (not just per-minute RPM)."""
    err = str(exc).lower()
    return (
        "per day" in err
        or "daily" in err
        or "generate_content_free_tier_requests" in err
        or "free_tier_requests" in err
    )


def _is_rate_limit(exc: Exception) -> bool:
    err = str(exc).lower()
    return (
        "429" in err
        or "resource_exhausted" in err
        or "resourceexhausted" in err
        or "quota" in err
        or "rate limit" in err
    )


def _is_not_found(exc: Exception) -> bool:
    err = str(exc).lower()
    return "404" in err or "not found" in err or "not supported" in err


def _is_auth_error(exc: Exception) -> bool:
    err = str(exc).lower()
    return (
        "401" in err
        or "unauthenticated" in err
        or "invalid_api_key" in err
        or "api_key_invalid" in err
        or "api key not valid" in err
    )


def _is_permission_error(exc: Exception) -> bool:
    err = str(exc).lower()
    return (
        "403" in err
        or "permission_denied" in err
        or "permissiondenied" in err
        or "not been used" in err
        or "disabled" in err
    )


def _is_network_error(exc: Exception) -> bool:
    err = str(exc).lower()
    return (
        "connection" in err
        or "connect" in err
        or "getaddrinfo" in err
        or "timeout" in err
        or "name resolution" in err
        or "socket" in err
    )


def _is_server_error(exc: Exception) -> bool:
    err = str(exc).lower()
    return "500" in err or "502" in err or "503" in err or "504" in err or "internal" in err


def build_candidate_list(
    primary_model: str,
    api_key: Optional[str] = None,
    for_audio_input: bool = False,
) -> List[str]:
    """
    Build ordered candidate models:
    1. primary_model (if specified and compatible with modality)
    2. Dynamically discovered models for this account
    3. Documented default fallbacks if discovery is empty
    """
    clean_primary = primary_model.replace("models/", "").strip()
    candidates: List[str] = []

    if for_audio_input:
        # STRICT RULE: NEVER route audio input to a Text-to-Speech (TTS) model!
        is_tts = "-tts" in clean_primary.lower() or "tts" in clean_primary.lower().split("-")
        if clean_primary and not is_tts:
            candidates.append(clean_primary)
        else:
            logger.info(
                f"Model '{clean_primary}' is a TTS model and cannot accept audio input. "
                "Routing audio input to 'gemini-1.5-flash'."
            )
            candidates.append("gemini-1.5-flash")

        if api_key:
            try:
                from app.utils.gemini_discovery import get_available_audio_input_models
                for m in get_available_audio_input_models(api_key):
                    if "-tts" not in m.lower() and m not in candidates:
                        candidates.append(m)
            except Exception as e:
                logger.debug(f"Could not get discovered audio input models: {e}")

        try:
            from app.utils.gemini_discovery import DOCUMENTED_DEFAULT_AUDIO_INPUT_MODELS
            for m in DOCUMENTED_DEFAULT_AUDIO_INPUT_MODELS:
                if m not in candidates:
                    candidates.append(m)
        except Exception:
            for m in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]:
                if m not in candidates:
                    candidates.append(m)

        return candidates

    # General text/vision: exclude TTS-only models
    if clean_primary and "-tts" not in clean_primary.lower():
        candidates.append(clean_primary)

    # Add discovered text models
    if api_key:
        discovered = get_available_text_models(api_key)
        for m in discovered:
            if "-tts" not in m.lower() and m not in candidates:
                candidates.append(m)

    # Add documented defaults as last resort
    for m in DOCUMENTED_DEFAULT_TEXT_MODELS:
        if m not in candidates:
            candidates.append(m)

    return candidates


# ---------------------------------------------------------------------------
# Public API: Text and Multimodal Generation
# ---------------------------------------------------------------------------

def call_gemini_with_retry(
    *,
    genai_module: Any,
    primary_model: str,
    parts: List[Any],
    history: Optional[List[Dict[str, Any]]] = None,
    generation_config: Any = None,
    system_instruction: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = _MAX_RETRIES,
) -> Tuple[str, str]:
    """
    Call Gemini with dynamic model discovery, pacing, and precise error handling.

    Returns: (response_text, model_used)

    Raises:
      - GeminiAuthError: 401
      - GeminiPermissionError: 403
      - GeminiRateLimitError: 429
      - GeminiNetworkError: connection / DNS failure
      - GeminiModelNotFoundError: 404 on all models
      - GeminiServerError: 500/503
    """
    # Detect if any part contains audio data (MIME audio/*)
    has_audio_input = any(
        (isinstance(p, dict) and str(p.get("mime_type", "")).startswith("audio/"))
        or (hasattr(p, "mime_type") and str(getattr(p, "mime_type", "")).startswith("audio/"))
        for p in parts
    )

    candidates = build_candidate_list(primary_model, api_key=api_key, for_audio_input=has_audio_input)
    history = history or []

    last_rate_limit_error: Optional[Exception] = None
    last_retry_after: float = 60.0
    last_is_daily: bool = False
    models_attempted: List[str] = []

    for model_name in candidates:
        models_attempted.append(model_name)
        attempt = 0

        while attempt <= max_retries:
            # ── Local RPM Pacing ──
            gemini_rate_limiter.wait_if_needed()

            try:
                model = genai_module.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction or "",
                    generation_config=generation_config,
                )
                chat = model.start_chat(history=history)
                response = chat.send_message(parts)
                gemini_rate_limiter.record_request()

                logger.info(f"Gemini response received using '{model_name}'.")
                return response.text.strip(), model_name

            except Exception as exc:
                # 1. Auth failure — fail immediately
                if _is_auth_error(exc):
                    logger.error(f"Gemini auth error on '{model_name}': {exc}")
                    raise GeminiAuthError(
                        "Your Gemini API key was rejected by Google (HTTP 401). "
                        "Please verify your key at https://aistudio.google.com/app/apikey"
                    ) from exc

                # 2. Permission failure — fail immediately
                if _is_permission_error(exc):
                    logger.error(f"Gemini permission error on '{model_name}': {exc}")
                    raise GeminiPermissionError(
                        "Access to Google Gemini API is denied (HTTP 403). "
                        "Please verify project permissions or billing settings in Google Cloud Console."
                    ) from exc

                # 3. Server error — fail immediately
                if _is_server_error(exc):
                    logger.error(f"Gemini server error on '{model_name}': {exc}")
                    raise GeminiServerError(
                        "Google's Gemini service returned a temporary server error. "
                        "Please try again in a few minutes."
                    ) from exc

                # 4. Network error — fail immediately
                if _is_network_error(exc):
                    logger.error(f"Gemini network error on '{model_name}': {exc}")
                    raise GeminiNetworkError(
                        f"Cannot connect to Google Gemini API: network error or timeout ({type(exc).__name__}). "
                        "Please check your internet connection."
                    ) from exc

                # 5. Audio modality mismatch (e.g. sent audio to a model that does not enable audio input)
                if "audio input modality is not enabled" in str(exc).lower():
                    logger.warning(
                        f"Model '{model_name}' rejected audio input modality: {exc}. "
                        "Advancing to next audio-capable candidate model..."
                    )
                    break

                # 6. Model not found (404) — try next discovered model
                if _is_not_found(exc):
                    logger.warning(f"Model '{model_name}' not available (404). Trying next model...")
                    break  # break inner retry loop → advance to next candidate model

                # 7. Rate limit / quota (429)
                if _is_rate_limit(exc):
                    last_rate_limit_error = exc
                    retry_after = _parse_retry_after(exc)
                    is_daily = _is_daily_quota_message(exc)

                    if retry_after is None:
                        retry_after = min(
                            _BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 5),
                            _MAX_BACKOFF,
                        )

                    if retry_after >= _DAILY_QUOTA_THRESHOLD_SECS:
                        is_daily = True

                    last_retry_after = retry_after
                    last_is_daily = is_daily

                    if is_daily:
                        # Daily quota applies to all models on the key — stop immediately
                        logger.warning(
                            f"Gemini daily quota reached on '{model_name}'. "
                            f"Retry in ~{retry_after/3600:.1f} hours."
                        )
                        raise GeminiRateLimitError(
                            f"Gemini's daily free-tier quota has been reached. "
                            f"Google restricts further requests for approximately {int(retry_after // 3600)} hour(s). "
                            "This is a Google account restriction and cannot be bypassed locally.",
                            retry_after=retry_after,
                            is_daily_quota=True,
                        )

                    # Per-minute limit: back-off and retry
                    attempt += 1
                    if attempt > max_retries:
                        logger.warning(f"RPM quota retries exhausted on '{model_name}'. Trying next model...")
                        break

                    logger.info(f"RPM limit on '{model_name}'. Waiting {retry_after:.1f}s before retry...")
                    time.sleep(retry_after)
                    continue

                # 7. Unclassified error
                logger.error(f"Unexpected Gemini error on '{model_name}': {type(exc).__name__}: {exc}")
                raise GeminiError(
                    f"Gemini API error ({type(exc).__name__}): {exc}"
                ) from exc

    # If quota was hit across models
    if last_rate_limit_error:
        raise GeminiRateLimitError(
            f"Gemini quota exceeded. Retry in ~{int(last_retry_after)}s.",
            retry_after=last_retry_after,
            is_daily_quota=last_is_daily,
        )

    # If all models returned 404
    raise GeminiModelNotFoundError(
        f"None of the candidate Gemini models were reachable on this account. "
        f"Attempted models: {', '.join(models_attempted)}. "
        "Please check available models in your Google AI Studio project."
    )


# ---------------------------------------------------------------------------
# Public API: Image Generation (Text-to-Image with Imagen)
# ---------------------------------------------------------------------------

def generate_image_with_gemini(
    *,
    api_key: str,
    prompt: str,
    model_name: Optional[str] = None,
    aspect_ratio: str = "1:1",
    number_of_images: int = 1,
) -> Image.Image:
    """
    Generate an image from a text prompt using Google's official Imagen models.
    Supports official Imagen 3 models (imagen-3.0-generate-002, etc.).

    Returns: PIL Image object.

    Raises:
      - GeminiMissingKeyError
      - GeminiAuthError
      - GeminiImageGenUnsupportedError: if Imagen is not enabled for this key
      - GeminiRateLimitError
      - GeminiError
    """
    if not api_key:
        raise GeminiMissingKeyError("GEMINI_API_KEY is required for image generation.")

    # 1. Determine model
    selected_imagen_model = model_name
    if not selected_imagen_model:
        available_image_models = get_available_image_models(api_key)
        if available_image_models:
            selected_imagen_model = available_image_models[0]
        else:
            selected_imagen_model = "imagen-3.0-generate-002"

    logger.info(f"Attempting image generation with model '{selected_imagen_model}'...")

    # Method A: Try google.generativeai SDK ImageGenerationModel
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        if hasattr(genai, "ImageGenerationModel"):
            imagen = genai.ImageGenerationModel(selected_imagen_model)
            gemini_rate_limiter.wait_if_needed()
            result = imagen.generate_images(
                prompt=prompt,
                number_of_images=number_of_images,
                aspect_ratio=aspect_ratio,
            )
            gemini_rate_limiter.record_request()
            if result.images:
                # result.images[0]._image is a PIL Image or bytes
                img_obj = result.images[0]._image
                if isinstance(img_obj, Image.Image):
                    return img_obj
                if hasattr(result.images[0], "_image_bytes"):
                    return Image.open(io.BytesIO(result.images[0]._image_bytes))
    except Exception as sdk_exc:
        logger.warning(f"SDK ImageGenerationModel failed ({sdk_exc}); trying REST API...")

    # Method B: Direct REST endpoint (official Google Generative Language predict endpoint)
    # POST https://generativelanguage.googleapis.com/v1beta/models/{model}:predict
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_imagen_model}:predict"
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "sampleCount": number_of_images,
            "aspectRatio": aspect_ratio,
        },
    }

    gemini_rate_limiter.wait_if_needed()
    try:
        resp = requests.post(
            url,
            json=payload,
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            timeout=45.0,
        )
        gemini_rate_limiter.record_request()
    except Exception as req_exc:
        if _is_network_error(req_exc):
            raise GeminiNetworkError(f"Network error during image generation: {req_exc}") from req_exc
        raise GeminiError(f"Failed to send image generation request: {req_exc}") from req_exc

    if resp.status_code == 200:
        data = resp.json()
        predictions = data.get("predictions", [])
        if predictions:
            b64_str = predictions[0].get("bytesBase64Encoded")
            if b64_str:
                img_bytes = base64.b64decode(b64_str)
                return Image.open(io.BytesIO(img_bytes))

    # Error handling for image generation
    status = resp.status_code
    body = resp.text
    body_lower = body.lower()

    if status in (401, 400) and ("api key" in body_lower or "invalid" in body_lower):
        raise GeminiAuthError("API key rejected during image generation.")
    if status == 403:
        raise GeminiImageGenUnsupportedError(
            "Image generation is not enabled for your Gemini project or region (HTTP 403). "
            "Google's Imagen 3 requires explicit access in Google AI Studio. "
            "Note: Text and image analysis remain fully operational."
        )
    if status == 404 or "not found" in body_lower:
        raise GeminiImageGenUnsupportedError(
            f"Image generation model '{selected_imagen_model}' is not available for this account. "
            "Google's free-tier Gemini API does not include Imagen 3 image generation by default. "
            "Note: Image understanding (uploading images to analyze) is fully functional."
        )
    if status == 429:
        is_daily = "per day" in body_lower or "daily" in body_lower
        raise GeminiRateLimitError(
            "Image generation quota exceeded.",
            retry_after=60.0,
            is_daily_quota=is_daily,
        )

    raise GeminiError(f"Image generation failed (HTTP {status}): {body[:200]}")
