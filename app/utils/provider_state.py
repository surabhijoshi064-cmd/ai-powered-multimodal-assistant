"""
Lightweight state tracking bridge delegating to services.quota_manager.

Maintains backward-compatibility for app/providers and app/services modules.
"""
from typing import Optional
from services.quota_manager import quota_manager


def mark_gemini_daily_quota_exhausted(retry_after_seconds: float = 3600.0) -> None:
    quota_manager.mark_daily_quota(retry_after_seconds=retry_after_seconds)


def clear_gemini_daily_quota_flag() -> None:
    quota_manager.clear_daily_quota()


def is_gemini_daily_quota_exhausted() -> bool:
    return quota_manager.is_daily_quota_active()


def get_quota_reset_eta_hours() -> float:
    secs = quota_manager.get_remaining_seconds()
    if secs is None:
        return 0.0
    return round(secs / 3600, 1)


def set_active_provider(name: str) -> None:
    quota_manager.set_active_provider(name)


def get_active_provider() -> Optional[str]:
    return quota_manager.get_active_provider()


def get_last_successful_provider() -> Optional[str]:
    return quota_manager.get_last_successful_provider()


def get_provider_status_snapshot() -> dict:
    summary = quota_manager.get_status_summary()
    return {
        "gemini_daily_quota_exhausted": summary["gemini_daily_quota_exhausted"],
        "gemini_quota_reset_eta_hours": get_quota_reset_eta_hours(),
        "active_provider": summary["active_provider"],
        "last_successful_provider": summary["last_successful_provider"],
    }
