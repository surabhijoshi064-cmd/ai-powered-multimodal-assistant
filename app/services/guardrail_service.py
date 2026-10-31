import re
from typing import Tuple, List
from app.config import settings
from app.utils.logging_config import setup_logging

logger = setup_logging(settings.LOG_LEVEL)

# Common prompt injection and jailbreak signatures
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+rules", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(DAN|unrestricted|in\s+developer\s+mode)", re.IGNORECASE),
    re.compile(r"(reveal|print|show|output)\s+(your\s+)?(system\s+prompt|initial\s+instructions)", re.IGNORECASE),
    re.compile(r"bypass\s+(safety|content\s+filter)", re.IGNORECASE),
]


class GuardrailService:
    @staticmethod
    def sanitize_user_input(text: str) -> str:
        """Sanitizes control characters and trims excessive whitespace."""
        # Remove null bytes or unusual control characters while keeping standard whitespace
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return sanitized.strip()

    @classmethod
    def check_prompt_injection(cls, text: str) -> Tuple[bool, str]:
        """
        Inspects text for prompt injection and override attempts.
        Returns (is_suspicious, reason).
        """
        for pattern in INJECTION_PATTERNS:
            if pattern.search(text):
                logger.warning(f"Potential prompt injection detected: {pattern.pattern}")
                return True, "Potential prompt injection or instruction override attempt detected."
        return False, ""

    @staticmethod
    def frame_untrusted_context(content: str, source_label: str = "Retrieved Document") -> str:
        """
        Wraps external/retrieved document text inside safe delimiter tags
        to prevent untrusted content from overriding system rules.
        """
        # Neutralize any attempts inside the document to close the tag
        safe_content = content.replace("</untrusted_context>", "[untrusted_context_end]")
        return (
            f"--- BEGIN {source_label} ---\n"
            f"<untrusted_context>\n"
            f"{safe_content}\n"
            f"</untrusted_context>\n"
            f"--- END {source_label} ---"
        )
