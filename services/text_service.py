"""
Text Generation Service for MindfulAI.

Coordinates prompt assembly, conversational context tracking, and response formatting.
"""
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are MindfulAI, an empathetic, supportive, and highly capable AI assistant. "
    "Provide compassionate listening, stress-reduction techniques (like box breathing or 5-4-3-2-1 grounding) "
    "when users express anxiety, and accurate, high-quality technical assistance for code and documents. "
    "Maintain clear ethical boundaries: you are an AI support tool, not a doctor or licensed therapist. "
    "When asked to refactor, edit, or explain code/documents, provide clear, production-ready outputs "
    "enclosed in appropriate markdown code fences."
)


class TextService:
    """Manages text prompt assembly and multi-turn message formatting."""

    @staticmethod
    def build_system_prompt(custom_prompt: Optional[str] = None) -> str:
        return custom_prompt or SYSTEM_PROMPT

    @staticmethod
    def format_history(
        messages_history: List[Dict[str, str]],
        current_prompt: str,
        doc_text: Optional[str] = None,
        doc_name: Optional[str] = None,
        max_turns: int = 12,
    ) -> List[Dict[str, str]]:
        """
        Assemble recent conversation history and prepend attached document content if present.
        """
        formatted = list(messages_history[-max_turns:]) if len(messages_history) > max_turns else list(messages_history)

        full_prompt = current_prompt.strip()
        if doc_text and doc_name:
            doc_header = f"[Attached File: {doc_name}]\n```\n{doc_text[:12000]}\n```\n\n"
            full_prompt = f"{doc_header}{full_prompt}" if full_prompt else f"{doc_header}Please analyze or edit this file."

        if full_prompt:
            formatted.append({"role": "user", "content": full_prompt})

        return formatted

