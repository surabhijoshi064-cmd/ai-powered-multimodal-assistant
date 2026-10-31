"""
Text-to-Speech (TTS) Generation Service for MindfulAI.

Responsibilities:
  - Generate speech audio from AI text responses.
  - Supports browser-based Web Speech API (zero latency, zero quota consumption).
  - Prepares safe text strings for audio synthesis by stripping markdown fences and URLs.
  - Provides downloadable audio where server-side audio generation is configured.
"""
import re
import urllib.parse
from typing import Optional


class TTSService:
    """Manages audio speech generation for assistant responses."""

    @staticmethod
    def clean_text_for_speech(text: str, max_chars: int = 1500) -> str:
        """Strip markdown fences, code blocks, bold/italics, and URLs for clear speech."""
        # Replace fenced code blocks with readable placeholder
        clean = re.sub(r"```.*?```", " code block omitted ", text, flags=re.DOTALL)
        # Remove inline code backticks
        clean = re.sub(r"`.*?`", "", clean)
        # Remove markdown formatting characters: * # _ ~ > [ ]
        clean = re.sub(r"[*#_`~>\[\]]", "", clean)
        # Remove URLs
        clean = re.sub(r"https?://\S+", "", clean)
        # Collapse whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean[:max_chars]

    @staticmethod
    def generate_browser_tts_js(text: str, voice_lang: str = "en-US", rate: float = 1.0) -> str:
        """
        Generate JavaScript to invoke the browser's native SpeechSynthesis API.
        Works across Chrome, Edge, Firefox, and Safari with zero cloud quota consumption.
        """
        clean_text = TTSService.clean_text_for_speech(text)
        safe_encoded = urllib.parse.quote(clean_text)

        return f"""
        <script>
        (function() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                const rawText = decodeURIComponent("{safe_encoded}");
                const utterance = new SpeechSynthesisUtterance(rawText);
                utterance.rate = {rate};
                utterance.pitch = 1.0;
                utterance.lang = '{voice_lang}';
                window.speechSynthesis.speak(utterance);
            }}
        }})();
        </script>
        """
