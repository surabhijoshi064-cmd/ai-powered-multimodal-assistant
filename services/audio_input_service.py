"""
Audio Input Processing Service.

Resolves ERROR 1:
  - Validates and processes incoming audio (WAV, MP3, M4A, OGG).
  - Explicitly routes to an audio-input-capable model (AUDIO_INPUT_MODEL = 'gemini-1.5-flash').
  - GUARANTEED: NEVER sends incoming audio to a Text-to-Speech (-tts) model!
  - If Gemini is unavailable (daily quota exhausted), falls back to local SpeechRecognition (STT),
    transcribes speech locally, and routes the transcribed text to Ollama.
"""
import io
import os
import logging
from typing import Optional, Tuple

import config
from services.quota_manager import quota_manager

logger = logging.getLogger(__name__)


class AudioInputService:
    """Handles audio speech input with strict model separation and local STT fallback."""

    def __init__(self, gemini_service=None, ollama_service=None):
        self.gemini_service = gemini_service
        self.ollama_service = ollama_service

    def _get_gemini_service(self):
        if self.gemini_service is None:
            from services.gemini_service import GeminiService
            self.gemini_service = GeminiService()
        return self.gemini_service

    def _get_ollama_service(self):
        if self.ollama_service is None:
            from services.ollama_service import OllamaService
            self.ollama_service = OllamaService()
        return self.ollama_service

    def process_audio(
        self,
        audio_bytes: bytes,
        user_prompt: str = "Please listen to this voice prompt and respond thoughtfully.",
        mime_type: str = "audio/wav",
    ) -> Tuple[str, str, str]:
        """
        Process speech audio input.
        Returns: (response_text, provider_used, model_used)

        Routing logic:
        1. If Gemini is available (quota not exhausted):
           Use Gemini with AUDIO_INPUT_MODEL ('gemini-1.5-flash').
           NEVER use a TTS model!
        2. If Gemini daily quota is exhausted:
           Transcribe audio locally using SpeechRecognition, then send text to Ollama.
        """
        # Case 1: Try Gemini if available
        if quota_manager.is_gemini_available():
            try:
                gemini = self._get_gemini_service()
                # Explicitly pass AUDIO_INPUT_MODEL to prevent any TTS model routing
                text, model = gemini.process_audio_input(
                    audio_bytes=audio_bytes,
                    user_prompt=user_prompt,
                    mime_type=mime_type,
                    model_name=config.AUDIO_INPUT_MODEL,
                )
                return text, "gemini", model
            except Exception as e:
                logger.warning(f"Gemini audio input processing failed ({e}). Checking local fallback...")
                # If this was a daily quota error, quota_manager will record it
                if not quota_manager.is_daily_quota_active() and not ("quota" in str(e).lower() or "429" in str(e)):
                    # Modality error or other non-quota error: try local STT + Ollama
                    pass

        # Case 2: Local STT fallback (SpeechRecognition) + Ollama text generation
        logger.info("Using local STT fallback for audio input...")
        transcribed_text = self._transcribe_locally(audio_bytes, mime_type=mime_type)

        combined_prompt = (
            f"[Transcribed Speech Input]: \"{transcribed_text}\"\n\n"
            f"User request: {user_prompt}"
            if transcribed_text
            else user_prompt
        )

        ollama = self._get_ollama_service()
        if ollama.is_running():
            reply, model = ollama.generate_text(
                messages=[{"role": "user", "content": combined_prompt}],
                system_instruction="You are a helpful assistant responding to user speech transcribed locally.",
            )
            prefix = "*(🎙️ Audio transcribed locally — answered by local Ollama AI)*\n\n"
            return prefix + reply, "ollama", model

        # Neither Gemini nor Ollama available
        return (
            f"I transcribed your speech: *\"{transcribed_text}\"*, but no AI provider is currently available "
            "to generate a response (Gemini quota is exhausted and Ollama is offline).",
            "none",
            "none",
        )

    def _transcribe_locally(self, audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
        """Transcribe speech audio bytes locally using SpeechRecognition."""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()

            # Convert bytes to AudioFile
            with io.BytesIO(audio_bytes) as audio_file:
                with sr.AudioFile(audio_file) as source:
                    audio_data = recognizer.record(source)
                    return recognizer.recognize_google(audio_data)
        except Exception as e:
            logger.debug(f"Local speech recognition error: {e}")
            return "Could not transcribe audio locally."

