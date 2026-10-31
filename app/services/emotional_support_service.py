import re
from typing import Tuple, Optional
from app.config import settings
from app.utils.logging_config import setup_logging

logger = setup_logging(settings.LOG_LEVEL)

# Crisis keywords indicating immediate danger, self-harm, or suicide
CRISIS_PATTERNS = [
    re.compile(r"\b(suicide|kill\s+myself|end\s+my\s+life|want\s+to\s+die|self[- ]harm|hanging\s+myself|overdose)\b", re.IGNORECASE),
    re.compile(r"\b(can'?t\s+go\s+on\s+anymore|better\s+off\s+dead|no\s+reason\s+to\s+live)\b", re.IGNORECASE),
]

CRISIS_RESPONSE_TEMPLATE = (
    "I am deeply concerned about what you are going through, and I want you to know that you are not alone. "
    "Because your safety and life are precious, please connect with someone who can support you right now:\n\n"
    "🆘 **Immediate Help & Crisis Resources (Confidential & 24/7):**\n"
    "- **Call or Text 988** (USA & Canada Suicide & Crisis Lifeline - free, confidential, 24/7)\n"
    "- **Text HOME to 741741** (Crisis Text Line)\n"
    "- **In the UK:** Call 111 (NHS) or 116 123 (Samaritans)\n"
    "- **In India:** Call 14416 (Tele-MANAS) or 9152987821 (KIRAN)\n"
    "- **International Resources:** Visit [findahelpline.com](https://findahelpline.com) to find immediate support in your country.\n\n"
    "Please reach out to one of these resources or a trusted friend, family member, or doctor immediately. People care and want to help you through this."
)


class EmotionalSupportService:
    @classmethod
    def evaluate_crisis_risk(cls, text: str) -> Tuple[bool, Optional[str]]:
        """
        Detects indications of acute crisis or self-harm risk.
        Returns (is_crisis, response_text).
        """
        for pattern in CRISIS_PATTERNS:
            if pattern.search(text):
                logger.warning("Crisis intent or self-harm pattern detected in user message.")
                return True, CRISIS_RESPONSE_TEMPLATE
        return False, None

    @staticmethod
    def get_grounding_technique() -> str:
        """Returns structured 5-4-3-2-1 sensory grounding technique."""
        return (
            "Here is the 5-4-3-2-1 grounding exercise to help bring you back to the present moment:\n"
            "- **5 things you can see** around you (e.g., a pen, the window, a book)\n"
            "- **4 things you can physically feel** (e.g., your feet on the floor, texture of your clothes)\n"
            "- **3 things you can hear** (e.g., a fan humming, distant traffic)\n"
            "- **2 things you can smell** (e.g., fresh air, tea or coffee)\n"
            "- **1 positive thought or thing you can taste**\n"
            "Take a slow breath between each step. How does that feel?"
        )
