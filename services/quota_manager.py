"""
Dedicated API Quota Management System for MindfulAI.

Responsibilities:
  - Detect HTTP 429 and Gemini quota-related API errors.
  - Distinguish daily free-tier quota exhaustion from short-term (RPM) rate limits.
  - Track provider availability and prevent repeatedly calling Gemini while daily quota is active.
  - Automatically resume Gemini requests when provider-supplied reset time or configured cooldown expires.
  - Use provider-supplied reset information when available; never invent an exact reset time.
  - Provide human-readable status metadata for Streamlit UI notifications.
"""
import re
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

from config import QUOTA_DEFAULT_COOLDOWN_SECONDS, GEMINI_API_KEY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error Inspection Utilities
# ---------------------------------------------------------------------------

DAILY_QUOTA_KEYWORDS = (
    "per day",
    "daily",
    "generate_content_free_tier_requests",
    "free_tier_requests",
    "day limit",
    "quota_exceeded_daily",
)

RATE_LIMIT_KEYWORDS = (
    "429",
    "resource_exhausted",
    "resourceexhausted",
    "quota",
    "rate limit",
    "rate_limit",
)


def is_rate_limit_error(exc: Exception) -> bool:
    """Return True if exception indicates any 429 or quota limit."""
    err = str(exc).lower()
    return any(kw in err for kw in RATE_LIMIT_KEYWORDS)


def is_daily_quota_error(exc: Exception) -> bool:
    """
    Distinguish daily quota exhaustion (20 RPD) from transient per-minute (RPM) limits.
    """
    err = str(exc).lower()
    return any(kw in err for kw in DAILY_QUOTA_KEYWORDS)


def parse_retry_after(exc: Exception) -> Optional[float]:
    """
    Extract provider-supplied retry delay from Gemini exception details or message.
    Returns float seconds if found, or None if provider gave no delay.
    Never invents an arbitrary time when metadata is absent.
    """
    # 1. Inspect protobuf / gRPC error details if present
    try:
        details = getattr(exc, "details", None)
        if callable(details):
            for detail in details():
                if hasattr(detail, "retry_delay"):
                    rd = detail.retry_delay
                    secs = rd.seconds + getattr(rd, "nanos", 0) / 1e9
                    if secs > 0:
                        return float(secs)
    except Exception:
        pass

    exc_str = str(exc)

    # 2. Hours and minutes pattern: "17h 30m" or "4 hours"
    hours_match = re.search(r"(\d+)\s*h(?:our)?s?\s*(\d+)?\s*m?", exc_str, re.I)
    if hours_match:
        hours = int(hours_match.group(1))
        mins = int(hours_match.group(2) or 0)
        return float(hours * 3600 + mins * 60)

    # 3. Seconds pattern: "retry in 60s" or "retry_delay: 45"
    sec_match = re.search(r"retry[_\s-]*(?:after|in|delay)[:\s]+(\d+(?:\.\d+)?)\s*s?", exc_str, re.I)
    if sec_match:
        return float(sec_match.group(1))

    return None


# ---------------------------------------------------------------------------
# QuotaManager Class
# ---------------------------------------------------------------------------

class QuotaManager:
    """
    Thread-safe session-wide API quota tracker.
    Prevents hammering Gemini when daily quota is exhausted and coordinates
    automatic failover to Ollama.
    """

    def __init__(self, default_cooldown_seconds: float = QUOTA_DEFAULT_COOLDOWN_SECONDS):
        self._lock = threading.RLock()
        self.default_cooldown_seconds = default_cooldown_seconds

        # State fields
        self._daily_quota_exhausted: bool = False
        self._exhausted_at: Optional[float] = None
        self._reset_at: Optional[float] = None
        self._reset_is_exact: bool = False
        self._last_error_message: str = ""

        # Provider tracking
        self._active_provider: str = "gemini"
        self._last_successful_provider: Optional[str] = None

    # ── Daily Quota State ────────────────────────────────────────────────

    def mark_daily_quota(
        self,
        retry_after_seconds: Optional[float] = None,
        error_message: str = "",
    ) -> None:
        """
        Mark that Gemini daily quota is exhausted.
        Uses provider-supplied reset seconds if available; otherwise uses configurable cooldown.
        Never invents an exact time when none was supplied.
        """
        with self._lock:
            now = time.time()
            self._daily_quota_exhausted = True
            self._exhausted_at = now
            self._last_error_message = error_message or "Daily free-tier request limit exceeded."

            if retry_after_seconds is not None and retry_after_seconds >= 0:
                self._reset_at = now + retry_after_seconds
                self._reset_is_exact = True
                logger.warning(
                    f"Gemini daily quota exhausted. Provider supplied reset delay: ~{retry_after_seconds/3600:.1f} hours."
                )
            else:
                self._reset_at = now + self.default_cooldown_seconds
                self._reset_is_exact = False
                logger.warning(
                    f"Gemini daily quota exhausted. Using configured cooldown of {self.default_cooldown_seconds/60:.0f} minutes."
                )

    def clear_daily_quota(self) -> None:
        """Explicitly clear the quota flag (e.g. manual reset button or after midnight)."""
        with self._lock:
            self._daily_quota_exhausted = False
            self._exhausted_at = None
            self._reset_at = None
            self._reset_is_exact = False
            self._last_error_message = ""
            logger.info("Gemini daily quota flag cleared.")

    def is_daily_quota_active(self) -> bool:
        """
        Return True if daily quota restriction is currently active.
        Automatically recovers (clears flag) once the reset time or cooldown has elapsed.
        """
        with self._lock:
            if not self._daily_quota_exhausted:
                return False

            now = time.time()
            if self._reset_at is not None and now >= self._reset_at:
                # Cooldown / reset time has passed — auto-recover!
                logger.info("Gemini quota reset window elapsed — automatically resuming Gemini availability.")
                self.clear_daily_quota()
                return False

            return True

    def get_remaining_seconds(self) -> Optional[float]:
        """Return seconds remaining until cooldown expires, or None if not exhausted."""
        with self._lock:
            if not self.is_daily_quota_active():
                return None
            if self._reset_at is None:
                return None
            return max(0.0, self._reset_at - time.time())

    def get_reset_display(self) -> str:
        """
        Return user-facing reset ETA string without fabricating unsupported claims.
        """
        with self._lock:
            if not self.is_daily_quota_active():
                return "Quota available"

            remaining = self.get_remaining_seconds()
            if remaining is None or remaining <= 0:
                return "Cooldown expired — ready to resume"

            hrs = int(remaining // 3600)
            mins = int((remaining % 3600) // 60)

            if self._reset_is_exact:
                if hrs > 0:
                    return f"~{hrs}h {mins}m (provider estimate)"
                return f"~{mins}m (provider estimate)"
            else:
                if hrs > 0:
                    return f"~{hrs} hour(s) cooldown (until midnight reset)"
                return f"~{mins} minute(s) cooldown"

    # ── Provider Availability ────────────────────────────────────────────

    def is_gemini_available(self) -> bool:
        """
        Check whether Gemini can currently be called:
        1. API key is configured.
        2. Daily quota restriction is NOT active.
        """
        from config import GEMINI_API_KEY
        if not GEMINI_API_KEY:
            return False
        return not self.is_daily_quota_active()

    def set_active_provider(self, name: str) -> None:
        """Record which provider generated the most recent response."""
        with self._lock:
            self._active_provider = name
            if name in ("gemini", "ollama"):
                self._last_successful_provider = name

    def get_active_provider(self) -> str:
        with self._lock:
            return self._active_provider

    def get_last_successful_provider(self) -> Optional[str]:
        with self._lock:
            return self._last_successful_provider

    def get_status_summary(self) -> Dict[str, Any]:
        """Return snapshot for UI rendering."""
        with self._lock:
            is_exhausted = self.is_daily_quota_active()
            return {
                "gemini_daily_quota_exhausted": is_exhausted,
                "gemini_available": self.is_gemini_available(),
                "reset_display": self.get_reset_display(),
                "reset_is_exact": self._reset_is_exact,
                "remaining_seconds": self.get_remaining_seconds(),
                "active_provider": self._active_provider,
                "last_successful_provider": self._last_successful_provider,
                "last_error_message": self._last_error_message,
            }


# Singleton instance shared across services and UI
quota_manager = QuotaManager()
