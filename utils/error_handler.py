"""
Error Formatting and UI Notification Utilities for MindfulAI.

Renders user-friendly, non-technical error notifications and provider status pills.
Never exposes raw stack traces or internal secrets to standard users.
"""
from typing import Dict, Any, Optional
import streamlit as st


class ErrorHandler:
    """Renders professional Streamlit notifications and banners."""

    @staticmethod
    def render_daily_quota_banner(
        reset_display: str,
        ollama_active: bool,
        ollama_model: str,
        ollama_running: bool,
    ):
        """
        Renders the professional quota-exceeded notification required by Section 6.
        """
        if ollama_active and ollama_running:
            st.info(
                f"""
### ⏳ GEMINI DAILY QUOTA EXCEEDED
**Your Gemini API free-tier quota has been reached.**

✅ **Switching to your local AI assistant...**
- **Provider:** Ollama Local AI
- **Model:** `{ollama_model}`
- **Status:** Connected & Ready
- **Gemini Reset:** {reset_display}

*Your conversation continues uninterrupted using your local computer resources.*
""",
                icon="🔄",
            )
        else:
            st.warning(
                f"""
### ⏳ GEMINI DAILY QUOTA EXCEEDED
**Your Gemini API free-tier quota has been reached.**

- **Gemini Reset:** {reset_display}
- **Local Fallback:** Ollama is not running.

To activate your local offline assistant:
1. Open Windows PowerShell
2. Run: `ollama serve`
3. Download a model: `ollama pull {ollama_model}`
""",
                icon="⚠️",
            )

    @staticmethod
    def render_provider_status_pills(
        gemini_connected: bool,
        gemini_daily_quota: bool,
        ollama_enabled: bool,
        ollama_running: bool,
        ollama_model: str,
        active_provider: str,
        masked_gemini_key: str = "",
    ):
        """
        Render dual AI provider status indicators adhering to Section 9.
        """
        # Gemini pill
        if not gemini_connected:
            gemini_html = '<span class="status-pill status-pill-offline">🔴 GEMINI AI — Key Required</span>'
        elif gemini_daily_quota:
            gemini_html = '<span class="status-pill status-pill-offline">⏳ GEMINI AI — Daily Quota Exceeded</span>'
        else:
            key_tag = f" ({masked_gemini_key})" if masked_gemini_key else ""
            gemini_html = f'<span class="status-pill status-pill-online">🟢 GEMINI AI — Connected{key_tag}</span>'

        # Ollama pill
        if not ollama_enabled:
            ollama_html = '<span class="status-pill" style="background:#f1f5f9;color:#64748b;border:1px solid #e2e8f0;">⚪ OLLAMA AI — Disabled</span>'
        elif ollama_running:
            if active_provider == "ollama":
                ollama_html = f'<span class="status-pill status-pill-online" style="background:#eff6ff;color:#1e40af;border:1px solid #bfdbfe;">🟢 OLLAMA AI — Local Fallback Active ({ollama_model})</span>'
            else:
                ollama_html = f'<span class="status-pill status-pill-online">🟢 OLLAMA AI — Ready ({ollama_model})</span>'
        else:
            ollama_html = '<span class="status-pill status-pill-offline">🔴 OLLAMA AI — Not Running</span>'

        st.markdown(gemini_html, unsafe_allow_html=True)
        st.markdown(ollama_html, unsafe_allow_html=True)

