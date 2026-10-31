"""
Official Google Gemini Model Discovery and Metadata Engine.

Uses official Google Gemini API:
  - SDK: google.generativeai.list_models()
  - REST: GET https://generativelanguage.googleapis.com/v1beta/models?key={api_key}

Extracts metadata (name, displayName, supportedGenerationMethods) and categorizes models:
  - Text / Multimodal Content Generation (generateContent)
  - Image Generation (imageGeneration / Imagen)
  - Embeddings (embedContent)

Implements caching, pagination, and strict error classification.
Never converts auth, quota, or network errors into false 'no models available'.
"""
import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import requests

logger = logging.getLogger(__name__)

# Fallback models documented by Google if discovery cannot reach network
DOCUMENTED_DEFAULT_TEXT_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
]

DOCUMENTED_DEFAULT_IMAGE_MODELS = [
    "imagen-3.0-generate-002",
]

DOCUMENTED_DEFAULT_AUDIO_INPUT_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
]

DOCUMENTED_DEFAULT_TTS_MODELS = [
    "gemini-2.5-flash-preview-tts",
]


@dataclass
class DiscoveredModel:
    """Represents a discovered Gemini model and its capabilities."""
    name: str                           # e.g. "models/gemini-1.5-flash"
    clean_name: str                     # e.g. "gemini-1.5-flash"
    display_name: str                   # e.g. "Gemini 1.5 Flash"
    description: str                    # Model description
    supported_generation_methods: List[str]
    supports_text: bool                 # 'generateContent' in methods, NOT a TTS-only model
    supports_vision: bool               # Multimodal generateContent
    supports_audio_input: bool          # Can process audio files/streams (never a TTS model)
    supports_tts: bool                  # Text-to-Speech output model
    supports_image_generation: bool     # 'imageGeneration' in methods or imagen in name
    supports_embedding: bool            # 'embedContent' in methods


@dataclass
class ModelDiscoveryResult:
    """Result of a model discovery query."""
    success: bool
    models: List[DiscoveredModel] = field(default_factory=list)
    text_models: List[str] = field(default_factory=list)        # clean_names for text/vision
    audio_input_models: List[str] = field(default_factory=list) # clean_names for audio input
    tts_models: List[str] = field(default_factory=list)         # clean_names for TTS output
    image_models: List[str] = field(default_factory=list)       # clean_names for image generation
    embedding_models: List[str] = field(default_factory=list)   # clean_names for embeddings
    error_type: Optional[str] = None    # missing_key, auth_failure, permission_denied, quota_exceeded, network_error, empty_list, unknown
    error_message: Optional[str] = None
    cached_at: float = field(default_factory=time.time)
    is_fallback: bool = False


# In-memory cache: api_key_hash -> (ModelDiscoveryResult, expiry_time)
_DISCOVERY_CACHE: Dict[str, Tuple[ModelDiscoveryResult, float]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _clean_model_name(raw_name: str) -> str:
    """Strip 'models/' prefix from model name if present."""
    if raw_name.startswith("models/"):
        return raw_name[len("models/"):]
    return raw_name


def _parse_model_object(model_data: Any) -> DiscoveredModel:
    """Parse either an SDK Model object or a REST JSON dict into DiscoveredModel."""
    if isinstance(model_data, dict):
        raw_name = model_data.get("name", "")
        display_name = model_data.get("displayName", "")
        description = model_data.get("description", "")
        methods = model_data.get("supportedGenerationMethods", [])
    else:
        raw_name = getattr(model_data, "name", "")
        display_name = getattr(model_data, "display_name", "") or getattr(model_data, "displayName", "")
        description = getattr(model_data, "description", "")
        methods = getattr(model_data, "supported_generation_methods", []) or getattr(model_data, "supportedGenerationMethods", [])

    clean_name = _clean_model_name(raw_name)
    clean_lower = clean_name.lower()
    methods_set = set(methods)

    # Detect specialized Text-to-Speech (TTS) output models
    # Models with '-tts' or 'preview-tts' generate speech and do NOT accept audio input!
    is_tts_model = (
        "-tts" in clean_lower
        or "preview-tts" in clean_lower
        or clean_lower.endswith("-tts")
        or "tts" in clean_lower.split("-")
    )
    supports_tts = is_tts_model or ("generateSpeech" in methods_set)

    # Text generation: general models supporting generateContent, excluding TTS-only models
    supports_text = ("generateContent" in methods_set) and not is_tts_model

    # Vision input: multimodal generateContent
    supports_vision = supports_text and ("gemini" in clean_lower)

    # Audio input: general Gemini models (1.5, 2.0, 2.5 general) that are explicitly NOT TTS models
    supports_audio_input = (
        supports_text
        and ("gemini" in clean_lower)
        and not is_tts_model
        and any(tag in clean_lower for tag in ("1.5", "2.0", "2.5", "flash", "pro"))
    )

    supports_image_gen = "imageGeneration" in methods_set or "imagen" in clean_lower
    supports_embedding = "embedContent" in methods_set or "embedText" in methods_set

    return DiscoveredModel(
        name=raw_name,
        clean_name=clean_name,
        display_name=display_name or clean_name,
        description=description or "",
        supported_generation_methods=list(methods),
        supports_text=supports_text,
        supports_vision=supports_vision,
        supports_audio_input=supports_audio_input,
        supports_tts=supports_tts,
        supports_image_generation=supports_image_gen,
        supports_embedding=supports_embedding,
    )


def _classify_http_error(status_code: int, response_text: str) -> Tuple[str, str]:
    """Classify an HTTP error response into an error_type and clean message."""
    resp_lower = response_text.lower()
    if status_code in (401, 400) and ("api key" in resp_lower or "invalid" in resp_lower or "unauthenticated" in resp_lower):
        return (
            "auth_failure",
            "Your Gemini API key was rejected by Google (HTTP 401). Please verify your key at https://aistudio.google.com/app/apikey"
        )
    if status_code == 403:
        if "not been used" in resp_lower or "disabled" in resp_lower:
            return (
                "permission_denied",
                "Google Generative Language API is not enabled for this project. Enable it in Google Cloud Console."
            )
        return (
            "permission_denied",
            "Access to Google Gemini API is denied (HTTP 403). Check your project permissions or billing settings."
        )
    if status_code == 429:
        if "per day" in resp_lower or "daily" in resp_lower:
            return (
                "quota_exceeded_daily",
                "Gemini daily quota exceeded. Google is temporarily restricting requests until the quota resets."
            )
        return (
            "quota_exceeded",
            "Gemini request rate limit exceeded (HTTP 429). Please wait a moment."
        )
    if status_code in (500, 502, 503, 504):
        return (
            "server_error",
            f"Google's Gemini service returned a server error (HTTP {status_code}). Please try again in a few minutes."
        )
    return (
        "unknown",
        f"Google Gemini API error (HTTP {status_code}): {response_text[:200]}"
    )


def discover_models_rest(api_key: str, timeout: float = 10.0) -> ModelDiscoveryResult:
    """
    Query the official Google Gemini API model list endpoint via REST:
    GET https://generativelanguage.googleapis.com/v1beta/models?key={api_key}
    Supports pagination with nextPageToken.
    """
    if not api_key:
        return ModelDiscoveryResult(
            success=False,
            error_type="missing_key",
            error_message="GEMINI_API_KEY is not set. Add it in the sidebar or .env file.",
        )

    url = "https://generativelanguage.googleapis.com/v1beta/models"
    headers = {"Content-Type": "application/json"}
    all_raw_models: List[Dict[str, Any]] = []
    page_token: Optional[str] = None

    try:
        while True:
            params: Dict[str, Any] = {"key": api_key, "pageSize": 100}
            if page_token:
                params["pageToken"] = page_token

            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code != 200:
                err_type, err_msg = _classify_http_error(resp.status_code, resp.text)
                logger.error(f"Model discovery REST error ({resp.status_code}): {err_msg}")
                return ModelDiscoveryResult(
                    success=False,
                    error_type=err_type,
                    error_message=err_msg,
                )

            data = resp.json()
            models_list = data.get("models", [])
            all_raw_models.extend(models_list)

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as conn_err:
        logger.warning(f"Model discovery network error: {conn_err}")
        return ModelDiscoveryResult(
            success=False,
            error_type="network_error",
            error_message=f"Unable to connect to Google Gemini API: network error or timeout ({type(conn_err).__name__}).",
            text_models=list(DOCUMENTED_DEFAULT_TEXT_MODELS),
            image_models=list(DOCUMENTED_DEFAULT_IMAGE_MODELS),
            is_fallback=True,
        )
    except Exception as exc:
        logger.error(f"Model discovery unexpected error: {exc}")
        return ModelDiscoveryResult(
            success=False,
            error_type="unknown",
            error_message=f"Failed to query model list ({type(exc).__name__}): {exc}",
        )

    # Process models
    parsed_models: List[DiscoveredModel] = []
    text_models: List[str] = []
    audio_input_models: List[str] = []
    tts_models: List[str] = []
    image_models: List[str] = []
    embedding_models: List[str] = []

    for item in all_raw_models:
        model = _parse_model_object(item)
        parsed_models.append(model)

        if model.supports_text:
            text_models.append(model.clean_name)
        if model.supports_audio_input:
            audio_input_models.append(model.clean_name)
        if model.supports_tts:
            tts_models.append(model.clean_name)
        if model.supports_image_generation:
            image_models.append(model.clean_name)
        if model.supports_embedding:
            embedding_models.append(model.clean_name)

    if not text_models:
        logger.warning("API key is valid, but 0 models supporting 'generateContent' were returned.")
        return ModelDiscoveryResult(
            success=False,
            models=parsed_models,
            error_type="empty_list",
            error_message="Your Gemini API key is valid, but Google returned 0 models supporting text generation.",
            audio_input_models=audio_input_models,
            tts_models=tts_models,
        )

    logger.info(f"Discovered {len(text_models)} text models, {len(audio_input_models)} audio input models, {len(image_models)} image models.")
    return ModelDiscoveryResult(
        success=True,
        models=parsed_models,
        text_models=text_models,
        audio_input_models=audio_input_models,
        tts_models=tts_models,
        image_models=image_models,
        embedding_models=embedding_models,
    )


def discover_models_sdk(api_key: str) -> ModelDiscoveryResult:
    """
    Query models using the official google.generativeai SDK.
    """
    if not api_key:
        return ModelDiscoveryResult(
            success=False,
            error_type="missing_key",
            error_message="GEMINI_API_KEY is not set.",
        )

    try:
        import google.generativeai as genai
    except ImportError:
        logger.warning("google.generativeai SDK not available; falling back to REST discovery.")
        return discover_models_rest(api_key)

    try:
        genai.configure(api_key=api_key)
        sdk_models = list(genai.list_models())

        parsed_models: List[DiscoveredModel] = []
        text_models: List[str] = []
        audio_input_models: List[str] = []
        tts_models: List[str] = []
        image_models: List[str] = []
        embedding_models: List[str] = []

        for item in sdk_models:
            model = _parse_model_object(item)
            parsed_models.append(model)
            if model.supports_text:
                text_models.append(model.clean_name)
            if model.supports_audio_input:
                audio_input_models.append(model.clean_name)
            if model.supports_tts:
                tts_models.append(model.clean_name)
            if model.supports_image_generation:
                image_models.append(model.clean_name)
            if model.supports_embedding:
                embedding_models.append(model.clean_name)

        if not text_models:
            return ModelDiscoveryResult(
                success=False,
                models=parsed_models,
                error_type="empty_list",
                error_message="API key valid, but no text-generation models found.",
                audio_input_models=audio_input_models,
                tts_models=tts_models,
            )

        return ModelDiscoveryResult(
            success=True,
            models=parsed_models,
            text_models=text_models,
            audio_input_models=audio_input_models,
            tts_models=tts_models,
            image_models=image_models,
            embedding_models=embedding_models,
        )

    except Exception as exc:
        err_str = str(exc).lower()
        if "401" in err_str or "api_key" in err_str or "unauthenticated" in err_str:
            return ModelDiscoveryResult(
                success=False,
                error_type="auth_failure",
                error_message=f"Gemini API authentication failed: {exc}",
            )
        if "403" in err_str or "permission" in err_str:
            return ModelDiscoveryResult(
                success=False,
                error_type="permission_denied",
                error_message=f"Gemini API permission denied: {exc}",
            )
        if "429" in err_str or "quota" in err_str:
            return ModelDiscoveryResult(
                success=False,
                error_type="quota_exceeded",
                error_message=f"Gemini API quota exceeded during discovery: {exc}",
            )
        # For other exceptions (network, etc.), try REST fallback
        logger.warning(f"SDK list_models failed ({exc}); trying REST fallback...")
        return discover_models_rest(api_key)


def discover_models(api_key: str, force_refresh: bool = False) -> ModelDiscoveryResult:
    """
    Main entry point for model discovery with caching.
    Tries SDK first, then REST fallback.
    Returns ModelDiscoveryResult with categorized models and clear error types.
    """
    if not api_key:
        return ModelDiscoveryResult(
            success=False,
            error_type="missing_key",
            error_message="Gemini API key is required. Add it to .env or the sidebar.",
            text_models=list(DOCUMENTED_DEFAULT_TEXT_MODELS),
            image_models=list(DOCUMENTED_DEFAULT_IMAGE_MODELS),
            is_fallback=True,
        )

    # Check cache (keyed by first 8 chars of key for safety)
    cache_key = api_key[:12] if len(api_key) >= 12 else api_key
    now = time.time()

    if not force_refresh and cache_key in _DISCOVERY_CACHE:
        cached_result, expiry = _DISCOVERY_CACHE[cache_key]
        if now < expiry and cached_result.success:
            return cached_result

    # Run discovery: try SDK then REST
    result = discover_models_sdk(api_key)
    if not result.success and result.error_type in ("network_error", "unknown"):
        # REST fallback already tried in discover_models_sdk; if network failed, provide documented defaults
        result.text_models = list(DOCUMENTED_DEFAULT_TEXT_MODELS)
        result.audio_input_models = list(DOCUMENTED_DEFAULT_AUDIO_INPUT_MODELS)
        result.tts_models = list(DOCUMENTED_DEFAULT_TTS_MODELS)
        result.image_models = list(DOCUMENTED_DEFAULT_IMAGE_MODELS)
        result.is_fallback = True

    if result.success:
        _DISCOVERY_CACHE[cache_key] = (result, now + CACHE_TTL_SECONDS)

    return result


def get_available_text_models(api_key: str) -> List[str]:
    """Convenience helper returning ordered list of clean model names for text generation."""
    res = discover_models(api_key)
    if res.text_models:
        return res.text_models
    return list(DOCUMENTED_DEFAULT_TEXT_MODELS)


def get_available_audio_input_models(api_key: str) -> List[str]:
    """Convenience helper returning models that support audio input (excluding TTS models)."""
    res = discover_models(api_key)
    if res.audio_input_models:
        return res.audio_input_models
    return list(DOCUMENTED_DEFAULT_AUDIO_INPUT_MODELS)


def get_available_tts_models(api_key: str) -> List[str]:
    """Convenience helper returning models designed for text-to-speech output."""
    res = discover_models(api_key)
    if res.tts_models:
        return res.tts_models
    return list(DOCUMENTED_DEFAULT_TTS_MODELS)


def get_available_image_models(api_key: str) -> List[str]:
    """Convenience helper returning available image generation models."""
    res = discover_models(api_key)
    if res.image_models:
        return res.image_models
    return list(DOCUMENTED_DEFAULT_IMAGE_MODELS)

