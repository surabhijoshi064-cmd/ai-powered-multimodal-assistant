"""
Multimodal processing service — Gemini vision, audio transcription, document text extraction.
"""
import os
import io
from typing import Optional, Tuple
from app.config import settings
from app.utils.logging_config import setup_logging
from app.utils.gemini_client import (
    call_gemini_with_retry,
    generate_image_with_gemini,
    GeminiRateLimitError,
    GeminiAuthError,
    GeminiImageGenUnsupportedError,
)

logger = setup_logging(settings.LOG_LEVEL)


class MultimodalService:
    """Handles image analysis, audio transcription, and document text extraction."""

    # ── Image Analysis (Gemini Vision) ──────────────────────────────────
    def analyze_image(
        self,
        image_bytes: bytes,
        filename: str = "image.png",
        user_prompt: str = "Describe this image in detail.",
    ) -> str:
        """
        Analyze an image using Gemini multimodal vision.
        Falls back to a description placeholder if Gemini is not configured.
        """
        if not settings.GEMINI_API_KEY:
            logger.info("No GEMINI_API_KEY — returning placeholder image description.")
            return (
                f"[Mock Vision]: Received image '{filename}' "
                f"({len(image_bytes)} bytes). "
                "Configure GEMINI_API_KEY in .env to enable real image analysis."
            )

        try:
            import google.generativeai as genai
            from PIL import Image

            genai.configure(api_key=settings.GEMINI_API_KEY)
            img = Image.open(io.BytesIO(image_bytes))

            generation_config = genai.types.GenerationConfig(
                temperature=0.4,
                max_output_tokens=1024,
            )

            target_model = settings.GEMINI_MODEL
            if "-tts" in target_model.lower():
                target_model = "gemini-1.5-flash"

            text, used_model = call_gemini_with_retry(
                genai_module=genai,
                primary_model=target_model,
                parts=[user_prompt, img],
                generation_config=generation_config,
                system_instruction="You are a helpful image analysis assistant.",
            )
            logger.info(f"Gemini vision analysis complete for '{filename}' using {used_model}.")
            return text

        except GeminiRateLimitError as e:
            logger.warning(f"Vision analysis rate-limited: {e}")
            if e.is_daily_quota:
                return (
                    "Image analysis is temporarily unavailable — Gemini's daily quota has been reached. "
                    f"Please try again in ~{int(e.retry_after // 3600)} hour(s)."
                )
            return f"Image analysis paused — quota limit. Please wait ~{int(e.retry_after)}s and retry."

        except GeminiAuthError as e:
            logger.error(f"Vision analysis auth error: {e}")
            return "Image analysis failed: API key authentication error. Please check GEMINI_API_KEY."

        except Exception as e:
            logger.error(f"Gemini vision error for '{filename}': {e}")
            return f"Image analysis failed: {type(e).__name__}. Please check your Gemini configuration."

    # ── Text-to-Image Generation (Imagen) ──────────────────────────────
    def generate_image(self, prompt: str, output_path: Optional[str] = None) -> Tuple[Optional[bytes], str]:
        """
        Generate an image from a text prompt using Google Imagen 3.
        Returns: (image_bytes_or_None, status_or_error_message)
        """
        if not settings.GEMINI_API_KEY:
            return None, "GEMINI_API_KEY is not configured in .env."

        try:
            img = generate_image_with_gemini(
                api_key=settings.GEMINI_API_KEY,
                prompt=prompt,
            )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            if output_path:
                img.save(output_path, format="PNG")
                logger.info(f"Generated image saved to {output_path}")

            return img_bytes, "Image generated successfully."

        except GeminiImageGenUnsupportedError as e:
            logger.warning(f"Image generation unsupported: {e}")
            return None, str(e)
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return None, f"Image generation failed: {e}"

    # ── Audio Transcription (Speech-to-Text) ────────────────────────────
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe an audio file to text.
        Uses Gemini for transcription when available, falls back to SpeechRecognition.
        """
        if settings.GEMINI_API_KEY and os.environ.get("TESTING") != "true":
            return self._transcribe_with_gemini(audio_path)
        return self._transcribe_with_speech_recognition(audio_path)

    def _transcribe_with_gemini(self, audio_path: str) -> str:
        """Use Gemini to transcribe audio."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=settings.GEMINI_API_KEY)

            audio_file = genai.upload_file(audio_path)

            generation_config = genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=2048,
            )

            target_model = settings.GEMINI_MODEL
            if "-tts" in target_model.lower():
                target_model = "gemini-1.5-flash"

            text, used_model = call_gemini_with_retry(
                genai_module=genai,
                primary_model=target_model,
                parts=[
                    "Transcribe this audio accurately. Return only the transcription text, "
                    "no commentary or formatting.",
                    audio_file,
                ],
                generation_config=generation_config,
            )
            logger.info(f"Gemini audio transcription complete: {len(text)} chars using {used_model}.")
            return text if text else "[No speech detected in audio]"

        except (GeminiRateLimitError, GeminiAuthError) as e:
            logger.warning(f"Gemini transcription quota/auth error — falling back: {e}")
            return self._transcribe_with_speech_recognition(audio_path)
        except Exception as e:
            logger.error(f"Gemini transcription error: {e}")
            return self._transcribe_with_speech_recognition(audio_path)

    def _transcribe_with_speech_recognition(self, audio_path: str) -> str:
        """Offline fallback using SpeechRecognition (Google free API)."""
        try:
            import speech_recognition as sr

            recognizer = sr.Recognizer()
            ext = os.path.splitext(audio_path)[1].lower()

            if ext != ".wav":
                logger.info(
                    f"Audio file is {ext} — SpeechRecognition works best with WAV. "
                    "Attempting anyway..."
                )

            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)

            text = recognizer.recognize_google(audio_data)
            logger.info(f"SpeechRecognition transcription complete: {len(text)} chars.")
            return text

        except ImportError:
            return "[Transcription unavailable: SpeechRecognition not installed]"
        except Exception as e:
            logger.error(f"SpeechRecognition error: {e}")
            return f"[Transcription failed: {type(e).__name__}]"

    # ── Document Text Extraction ────────────────────────────────────────
    def extract_document_text(self, content: bytes, ext: str) -> str:
        """Extract text from a PDF or TXT file."""
        try:
            if ext == "pdf":
                import fitz  # PyMuPDF
                text_parts = []
                with fitz.open(stream=content, filetype="pdf") as doc:
                    for page in doc:
                        text_parts.append(page.get_text())
                return "\n".join(text_parts)

            elif ext == "txt":
                try:
                    return content.decode("utf-8")
                except UnicodeDecodeError:
                    return content.decode("latin-1", errors="replace")

            else:
                try:
                    return content.decode("utf-8")
                except UnicodeDecodeError:
                    return content.decode("latin-1", errors="replace")

        except Exception as e:
            logger.error(f"Document extraction error: {e}")
            return f"[Failed to extract text: {type(e).__name__}]"

    # ── Text-to-Speech (Gemini or gTTS) ─────────────────────────────────
    def text_to_speech(self, text: str, output_path: str) -> Optional[str]:
        """
        Generate an audio file from text.
        Returns the file path on success, None on failure.
        Uses browser-side Web Speech API by default (no backend audio generation needed).
        This method is provided as an optional server-side fallback.
        """
        try:
            from gtts import gTTS
            tts = gTTS(text=text[:5000], lang="en", slow=False)
            tts.save(output_path)
            logger.info(f"TTS audio saved to {output_path}")
            return output_path
        except ImportError:
            logger.info("gTTS not installed — TTS handled client-side via Web Speech API.")
            return None
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            return None

