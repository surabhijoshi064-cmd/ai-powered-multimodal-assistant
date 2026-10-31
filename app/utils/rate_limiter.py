"""
GeminiRateLimiter — Thread-safe request pacing for the Google Gemini free tier.

Free tier limit: 5 requests per minute (RPM).
Default safety margin: 4 RPM (configurable via GEMINI_RPM_LIMIT env var).
"""
import os
import time
import threading
import random
from collections import deque
from typing import Deque


class GeminiRateLimiter:
    """
    Tracks per-minute request timestamps and enforces a configurable RPM ceiling.

    Usage:
        limiter.wait_if_needed()   # call BEFORE sending request
        # ... send request ...
        limiter.record_request()   # call AFTER successful send
    """

    def __init__(self, requests_per_minute: int = 4, window_seconds: int = 60):
        self._rpm = requests_per_minute
        self._window = window_seconds
        self._timestamps: Deque[float] = deque()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wait_if_needed(self) -> None:
        """Block until it is safe to send the next request without exceeding RPM."""
        while True:
            with self._lock:
                self._purge_old()
                if len(self._timestamps) < self._rpm:
                    return  # slot available — caller may proceed
                # Oldest timestamp tells us when a slot will free up
                oldest = self._timestamps[0]
                wait_sec = (oldest + self._window) - time.monotonic()

            # Release lock while sleeping so other threads are not blocked
            if wait_sec > 0:
                # Add small jitter (0–2 s) to smooth bursts
                jitter = random.uniform(0, min(2.0, wait_sec * 0.1))
                time.sleep(wait_sec + jitter)

    def record_request(self) -> None:
        """Record a timestamp for a request that was just sent."""
        with self._lock:
            self._timestamps.append(time.monotonic())

    def seconds_until_next_slot(self) -> float:
        """Return estimated seconds until the next slot is free (0 if slot available now)."""
        with self._lock:
            self._purge_old()
            if len(self._timestamps) < self._rpm:
                return 0.0
            oldest = self._timestamps[0]
            return max(0.0, (oldest + self._window) - time.monotonic())

    def current_count(self) -> int:
        """Return the number of requests sent in the current rolling window."""
        with self._lock:
            self._purge_old()
            return len(self._timestamps)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _purge_old(self) -> None:
        """Remove timestamps older than the rolling window (must be called with lock held)."""
        cutoff = time.monotonic() - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()


# ---------------------------------------------------------------------------
# Singleton — shared across all modules in the same process
# ---------------------------------------------------------------------------
_rpm_limit = int(os.getenv("GEMINI_RPM_LIMIT", "4"))
gemini_rate_limiter = GeminiRateLimiter(requests_per_minute=_rpm_limit)

