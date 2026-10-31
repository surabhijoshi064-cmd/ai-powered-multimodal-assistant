"""
MindfulAI — Premium Futuristic 3D JARVIS-Style Multimodal AI Assistant & Image Studio

Identity & Branding: MindfulAI (Preserved)
Design Inspiration: Iron Man JARVIS • Holographic 3D Command Center • Sci-Fi Glassmorphism
Engines:
  - Primary AI: Google Gemini (Multimodal reasoning, vision, audio input, Imagen 3)
  - Secondary AI: Local Ollama (Offline chat fallback on Gemini quota exhaustion)
  - Image Engines: Google Imagen 3 (Cloud) + ComfyUI SD1.5/SDXL (Local)
"""
import os
import io
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import streamlit as st
from PIL import Image
from streamlit_mic_recorder import mic_recorder

import config
from services.quota_manager import quota_manager
from services.ai_router import ai_router, TaskIntent, TurnResult
from services.tts_service import TTSService
from services.image_generation_service import STYLE_PRESETS, ASPECT_RATIOS
from services.image_editing_service import EDIT_PRESETS
from services.image_storage_service import image_storage_service
from utils.session_manager import SessionManager
from utils.error_handler import ErrorHandler
from app.utils.gemini_discovery import (
    discover_models,
    DOCUMENTED_DEFAULT_TEXT_MODELS,
    DOCUMENTED_DEFAULT_IMAGE_MODELS,
)

# ---------------------------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MindfulAI | JARVIS 3D Multimodal Studio",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
SessionManager.initialize()

# Session State for Studio Navigation & Image Workspaces
if "studio_mode" not in st.session_state:
    st.session_state.studio_mode = "💬 AI Chat"
if "editor_source_bytes" not in st.session_state:
    st.session_state.editor_source_bytes = None
if "editor_edited_bytes" not in st.session_state:
    st.session_state.editor_edited_bytes = None
if "editor_instruction" not in st.session_state:
    st.session_state.editor_instruction = ""
if "studio_gen_history" not in st.session_state:
    st.session_state.studio_gen_history = []

# ---------------------------------------------------------------------------
# Futuristic 3D JARVIS Theme (Custom CSS)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Root Color Palette (JARVIS Deep Space) ── */
    :root {
        --jarvis-bg: #040714;
        --jarvis-navy: #0B132B;
        --jarvis-surface: #111D3B;
        --jarvis-surface-alt: #1C2541;
        --jarvis-cyan: #00F2FE;
        --jarvis-blue: #3B82F6;
        --jarvis-violet: #8B5CF6;
        --jarvis-purple: #A855F7;
        --jarvis-text-bright: #F8FAFC;
        --jarvis-text-dim: #94A3B8;
        --jarvis-glow: 0 0 20px rgba(0, 242, 254, 0.45);
        --jarvis-glow-violet: 0 0 25px rgba(139, 92, 246, 0.45);
    }

    /* ── Background & General Typography ── */
    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(0, 242, 254, 0.08) 0%, transparent 45%),
                    radial-gradient(circle at 90% 90%, rgba(139, 92, 246, 0.08) 0%, transparent 45%),
                    radial-gradient(circle at 50% 50%, rgba(11, 19, 43, 0.5) 0%, transparent 60%),
                    #040714 !important;
        color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Sidebar Styling ── */
    [data-testid="stSidebar"] {
        background: rgba(11, 19, 43, 0.94) !important;
        border-right: 1px solid rgba(0, 242, 254, 0.22) !important;
        backdrop-filter: blur(20px);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.6);
    }

    /* ── JARVIS Holographic Header ── */
    .jarvis-header-card {
        background: linear-gradient(135deg, rgba(11, 19, 43, 0.85) 0%, rgba(28, 37, 65, 0.75) 100%);
        border: 1px solid rgba(0, 242, 254, 0.35);
        border-radius: 20px;
        padding: 1.4rem 2rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.6), inset 0 1px 1px rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(16px);
        margin-bottom: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1.2rem;
    }

    .jarvis-title-group {
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .jarvis-app-name {
        font-size: 2.3rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #00F2FE 0%, #3B82F6 45%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .jarvis-tagline {
        color: #94A3B8;
        font-size: 0.88rem;
        margin-top: 3px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        font-weight: 600;
    }

    /* ── Telemetry HUD Pills ── */
    .hud-telemetry {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }

    .hud-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 13px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }

    .hud-online {
        background: rgba(34, 197, 94, 0.14);
        color: #4ade80;
        border-color: rgba(74, 222, 128, 0.35);
        box-shadow: 0 0 12px rgba(74, 222, 128, 0.2);
    }

    .hud-active-cyan {
        background: rgba(0, 242, 254, 0.14);
        color: #00F2FE;
        border-color: rgba(0, 242, 254, 0.4);
        box-shadow: 0 0 14px rgba(0, 242, 254, 0.25);
    }

    .hud-paused {
        background: rgba(245, 158, 11, 0.14);
        color: #fbbf24;
        border-color: rgba(251, 191, 36, 0.35);
        box-shadow: 0 0 12px rgba(251, 191, 36, 0.2);
    }

    .hud-offline {
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border-color: rgba(248, 113, 113, 0.3);
    }

    /* ── 3D Glass Cards ── */
    .glass-card-3d {
        background: linear-gradient(135deg, rgba(17, 29, 59, 0.8) 0%, rgba(28, 37, 65, 0.6) 100%);
        border: 1px solid rgba(0, 242, 254, 0.22);
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.55), inset 0 1px 1px rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(14px);
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .glass-card-3d:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 242, 254, 0.5);
        box-shadow: 0 15px 40px rgba(0, 242, 254, 0.15);
    }

    .card-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0.45rem;
        display: flex;
        align-items: center;
        gap: 9px;
    }

    .card-caption {
        color: #94A3B8;
        font-size: 0.86rem;
        line-height: 1.5;
    }

    /* ── Provider Attribution Pill ── */
    .prov-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 11px;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .prov-gemini {
        background: rgba(0, 242, 254, 0.16);
        color: #00F2FE;
        border: 1px solid rgba(0, 242, 254, 0.4);
    }

    .prov-ollama {
        background: rgba(139, 92, 246, 0.16);
        color: #A78BFA;
        border: 1px solid rgba(139, 92, 246, 0.4);
    }

    .prov-comfy {
        background: rgba(244, 63, 94, 0.16);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.4);
    }

    /* ── Streamlit Button Customization ── */
    .stButton>button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        transition: all 0.25s ease !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
    }

    .stButton>button:hover {
        border-color: rgba(0, 242, 254, 0.8) !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.3) !important;
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_code_blocks(markdown_text: str):
    import re
    pattern = r"```([a-zA-Z0-9_\-\+]*)\n(.*?)```"
    matches = re.findall(pattern, markdown_text, flags=re.DOTALL)
    return [(lang.strip() or "txt", code.strip()) for lang, code in matches]


def extract_file_content(file_bytes: bytes, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    if ext == "pdf":
        try:
            import pymupdf
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        except Exception as e:
            return f"[PDF parsing error: {e}]"
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="replace")


# ---------------------------------------------------------------------------
# Telemetry Checks
# ---------------------------------------------------------------------------
active_key = config.GEMINI_API_KEY
masked_key = f"{active_key[:4]}...{active_key[-4:]}" if len(active_key) > 8 else "NOT CONFIGURED"
is_daily_quota = quota_manager.is_daily_quota_active()

ollama_info = ai_router.ollama_service.get_status_info() if config.OLLAMA_ENABLED else {}
ollama_running = ollama_info.get("running", False)

from services.comfyui_service import ComfyUIService
comfy_svc = ComfyUIService()
comfy_running = comfy_svc.is_available()

# ---------------------------------------------------------------------------
# Sidebar: 3D Control Center & Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 MindfulAI")
    st.markdown("*JARVIS Multimodal Intelligence*")

    st.markdown("#### 🎛️ Navigation Studio")
    selected_studio = st.radio(
        "Studio Navigation",
        options=[
            "💬 AI Chat",
            "🎨 AI Image Studio",
            "🪄 Image Editor",
            "🎙️ Voice Assistant",
            "⚙️ Engine Matrix",
        ],
        index=[
            "💬 AI Chat",
            "🎨 AI Image Studio",
            "🪄 Image Editor",
            "🎙️ Voice Assistant",
            "⚙️ Engine Matrix",
        ].index(st.session_state.studio_mode),
        label_visibility="collapsed",
    )
    if selected_studio != st.session_state.studio_mode:
        st.session_state.studio_mode = selected_studio
        st.rerun()

    st.divider()

    # System Telemetry Panel
    st.markdown("#### 🛰️ Core Diagnostics")
    if is_daily_quota:
        st.markdown('<span class="hud-badge hud-paused">⚡ Gemini: Quota Paused</span>', unsafe_allow_html=True)
        st.caption(f"Cooldown reset in ~**{quota_manager.get_reset_display()}**")
        if st.button("🔓 Clear Daily Quota Lock", key="sb_clear_quota"):
            quota_manager.clear_daily_quota()
            st.rerun()
    elif active_key:
        st.markdown('<span class="hud-badge hud-online">⚡ Gemini: Operational</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="hud-badge hud-offline">⚡ Gemini: Key Required</span>', unsafe_allow_html=True)

    if ollama_running:
        st.markdown(f'<span class="hud-badge hud-online">🖥️ Ollama: Ready ({config.OLLAMA_TEXT_MODEL})</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="hud-badge hud-offline">🖥️ Ollama: Standby (Offline)</span>', unsafe_allow_html=True)

    if comfy_running:
        st.markdown('<span class="hud-badge hud-online">🎨 ComfyUI: Online (Port 8188)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="hud-badge hud-offline">🎨 ComfyUI: Standby</span>', unsafe_allow_html=True)

    st.divider()

    # Model Settings Expander
    discovery_res = discover_models(active_key) if active_key else None
    model_options = discovery_res.text_models if (discovery_res and discovery_res.text_models) else list(DOCUMENTED_DEFAULT_TEXT_MODELS)
    default_idx = model_options.index(config.TEXT_MODEL) if config.TEXT_MODEL in model_options else 0

    with st.expander("⚙️ AI Model Matrix", expanded=False):
        selected_gemini_model = st.selectbox(
            "Gemini Text & Vision Model",
            options=model_options,
            index=default_idx,
        )
        st.caption(f"🎙️ Audio Input: `{config.AUDIO_INPUT_MODEL}`")
        st.caption(f"🔊 Dedicated TTS: `{config.TTS_MODEL}`")
        st.caption(f"🎨 Image Model: `{config.IMAGEN_MODEL}`")

        if active_key and st.button("🔄 Refresh Models", key="sb_refresh_models"):
            discover_models(active_key, force_refresh=True)
            st.rerun()

    # Audio synthesis toggle
    st.markdown("#### 🔊 Audio Synthesis")
    st.session_state.tts_enabled = st.toggle(
        "Auto-Voice Output",
        value=st.session_state.tts_enabled,
        help="Spoken response playback via neural browser speech synthesis.",
    )

    st.divider()

    # Session Archive
    if st.button("➕ New Conversation", use_container_width=True, type="primary"):
        SessionManager.create_conversation()
        st.rerun()

    st.markdown("#### 💬 Session Archive")
    conv_items = list(st.session_state.conversations.items())
    if not conv_items:
        st.caption("No conversations recorded yet.")
    else:
        for cid, cdata in conv_items:
            col1, col2 = st.columns([0.82, 0.18])
            is_active = cid == st.session_state.current_conv_id
            btn_title = f"{'👉 ' if is_active else ''}{cdata['title'][:20]}"
            with col1:
                if st.button(btn_title, key=f"sel_{cid}", use_container_width=True):
                    st.session_state.current_conv_id = cid
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{cid}", help="Delete chat"):
                    SessionManager.delete_conversation(cid)
                    st.rerun()

# ---------------------------------------------------------------------------
# JARVIS Holographic Arc Reactor Top Banner
# ---------------------------------------------------------------------------
active_engine = quota_manager.get_active_provider().upper()
st.markdown(
    f"""
    <div class="jarvis-header-card">
        <div class="jarvis-title-group">
            <!-- Animated 3D Arc-Reactor SVG Orb -->
            <svg width="68" height="68" viewBox="0 0 100 100" style="filter: drop-shadow(0 0 10px #00F2FE); flex-shrink: 0;">
                <circle cx="50" cy="50" r="44" fill="none" stroke="rgba(0, 242, 254, 0.3)" stroke-width="2"/>
                <circle cx="50" cy="50" r="38" fill="none" stroke="#00F2FE" stroke-width="3" stroke-dasharray="12 6" transform="rotate(0 50 50)">
                    <animateTransform attributeName="transform" type="rotate" from="0 50 50" to="360 50 50" dur="12s" repeatCount="indefinite"/>
                </circle>
                <circle cx="50" cy="50" r="30" fill="none" stroke="#8B5CF6" stroke-width="2.5" stroke-dasharray="8 4">
                    <animateTransform attributeName="transform" type="rotate" from="360 50 50" to="0 50 50" dur="8s" repeatCount="indefinite"/>
                </circle>
                <circle cx="50" cy="50" r="20" fill="none" stroke="#3B82F6" stroke-width="2"/>
                <circle cx="50" cy="50" r="12" fill="#00F2FE">
                    <animate attributeName="opacity" values="0.7;1;0.7" dur="2s" repeatCount="indefinite"/>
                    <animate attributeName="r" values="10;14;10" dur="2s" repeatCount="indefinite"/>
                </circle>
            </svg>
            <div>
                <div class="jarvis-app-name">🌿 MindfulAI</div>
                <div class="jarvis-tagline">JARVIS 3D Multimodal Assistant • Real Image Studio & Editor • Cloud & Local Engine</div>
            </div>
        </div>
        <div class="hud-telemetry">
            <span class="hud-badge hud-active-cyan">Active Core: {active_engine}</span>
            <span class="hud-badge {'hud-online' if bool(active_key) and not is_daily_quota else 'hud-paused'}">Gemini Cloud</span>
            <span class="hud-badge {'hud-online' if ollama_running else 'hud-offline'}">Ollama Local</span>
            <span class="hud-badge {'hud-online' if comfy_running else 'hud-offline'}">ComfyUI Studio</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quota Alert
if is_daily_quota:
    ErrorHandler.render_daily_quota_banner(
        reset_display=quota_manager.get_reset_display(),
        ollama_active=config.OLLAMA_ENABLED,
        ollama_model=config.OLLAMA_TEXT_MODEL,
        ollama_running=ollama_running,
    )


# ---------------------------------------------------------------------------
# Turn Execution Pipeline (Universal Multimodal Pipeline)
# ---------------------------------------------------------------------------
def execute_turn(
    prompt: str,
    img_obj: Optional[Image.Image] = None,
    audio_bytes: Optional[bytes] = None,
    doc_text: Optional[str] = None,
    doc_name: Optional[str] = None,
    aspect_ratio: str = "1:1",
    style: str = "Cinematic",
):
    if st.session_state.get("_request_in_flight", False):
        return
    st.session_state["_request_in_flight"] = True

    try:
        display_user = prompt
        if doc_name:
            display_user = f"📎 *Attached [{doc_name}]*\n\n{prompt}"
        elif img_obj is not None:
            display_user = f"🖼️ *[Image Attached]*\n\n{prompt}"
        elif audio_bytes is not None:
            display_user = f"🎤 *[Voice Prompt]*\n\n{prompt}"

        # First turn auto-title
        curr_msgs = SessionManager.get_current_messages()
        if not curr_msgs and prompt:
            cid = st.session_state.current_conv_id
            st.session_state.conversations[cid]["title"] = prompt[:24] + ("..." if len(prompt) > 24 else "")

        SessionManager.append_message("user", display_user)

        with st.chat_message("user", avatar="👤"):
            st.markdown(display_user)

        with st.chat_message("assistant", avatar="🌿"):
            with st.spinner("MindfulAI is synthesizing response..."):
                result: TurnResult = ai_router.route_turn(
                    prompt=prompt,
                    messages_history=curr_msgs,
                    attached_image=img_obj,
                    attached_audio_bytes=audio_bytes,
                    attached_doc_text=doc_text,
                    attached_doc_name=doc_name,
                    aspect_ratio=aspect_ratio,
                    style=style,
                )

            if not result.success and result.error_title:
                st.error(f"**{result.error_title}**")
                st.warning(result.error_message or result.response_text)
                return

            # Attribution badge
            if result.provider == "gemini":
                st.markdown(f'<span class="prov-pill prov-gemini">⚡ Gemini Cloud · {result.model_used}</span>', unsafe_allow_html=True)
            elif result.provider == "ollama":
                st.markdown(f'<span class="prov-pill prov-ollama">🖥️ Ollama Local · {result.model_used}</span>', unsafe_allow_html=True)
            elif result.provider == "comfyui":
                st.markdown(f'<span class="prov-pill prov-comfy">🎨 ComfyUI Local · {result.model_used}</span>', unsafe_allow_html=True)

            st.markdown(result.response_text)

            # Display generated or edited images
            display_img = result.edited_image_bytes or result.generated_image_bytes
            if display_img:
                st.image(display_img, caption="MindfulAI Visual Output", use_container_width=True)

            SessionManager.append_message(
                role="assistant",
                content=result.response_text,
                provider=result.provider,
                model=result.model_used,
                image_bytes=display_img,
            )

            if st.session_state.get("tts_enabled", False):
                st.session_state["_auto_tts_text"] = result.response_text

    finally:
        st.session_state["_request_in_flight"] = False
    st.rerun()


# ===========================================================================
# MODE 1: 💬 AI CHAT (NEURAL CONVERSATIONAL ASSISTANT)
# ===========================================================================
if st.session_state.studio_mode == "💬 AI Chat":
    current_messages = SessionManager.get_current_messages()

    if not current_messages:
        st.markdown(
            """
            <div class="glass-card-3d">
                <div class="card-title">✨ JARVIS Holographic Command Center</div>
                <div class="card-caption">
                    Full-spectrum conversational AI ready for technical reasoning, complex code generation & debugging,
                    mathematical proofs, multimodal vision understanding, and real-time speech transcription.<br>
                    Select a starter directive below or enter your own command in the dock.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        STARTER_PROMPTS = [
            "🎨 Generate a realistic lion standing on a mountain peak at sunrise",
            "🤖 Create a 3D robot designed for deep space exploration",
            "🐍 Write an async Python rate-limited client with exponential backoff",
            "🧠 Explain quantum superposition with mathematical formulation",
            "📄 Analyze and extract strategic insights from my attached document",
            "🪄 Edit this image to add sunglasses and cinematic warm lighting",
        ]
        chip_cols = st.columns(3)
        for i, p in enumerate(STARTER_PROMPTS):
            with chip_cols[i % 3]:
                if st.button(p, key=f"starter_chip_{i}", use_container_width=True):
                    st.session_state["_pending_prompt"] = p
                    st.rerun()

    # Render Conversation Messages
    for msg_idx, msg in enumerate(current_messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        avatar = "👤" if role == "user" else "🌿"

        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                prov = msg.get("provider", "gemini")
                model = msg.get("model", "")
                if prov == "gemini":
                    st.markdown(f'<span class="prov-pill prov-gemini">⚡ Gemini Cloud · {model}</span>', unsafe_allow_html=True)
                elif prov == "ollama":
                    st.markdown(f'<span class="prov-pill prov-ollama">🖥️ Ollama Local · {model}</span>', unsafe_allow_html=True)
                elif prov == "comfyui":
                    st.markdown(f'<span class="prov-pill prov-comfy">🎨 ComfyUI Local · {model}</span>', unsafe_allow_html=True)

            st.markdown(content)

            if msg.get("image_bytes"):
                st.image(msg["image_bytes"], caption="MindfulAI Rendered Output", use_container_width=True)
                c_btn1, c_btn2 = st.columns([0.5, 0.5])
                with c_btn1:
                    st.download_button(
                        label="📥 Download Image (.png)",
                        data=msg["image_bytes"],
                        file_name=f"mindfulai_render_{msg_idx+1}.png",
                        mime="image/png",
                        key=f"dl_chat_img_{msg_idx}",
                    )
                with c_btn2:
                    if st.button("🪄 Open in Image Editor", key=f"open_editor_{msg_idx}"):
                        st.session_state.editor_source_bytes = msg["image_bytes"]
                        st.session_state.studio_mode = "🪄 Image Editor"
                        st.rerun()

            if role == "assistant":
                act_c1, act_c2 = st.columns([0.2, 0.8])
                with act_c1:
                    if st.button("🔊 Voice", key=f"chat_tts_{msg_idx}"):
                        st.components.v1.html(TTSService.generate_browser_tts_js(content), height=0, width=0)

                code_blocks = extract_code_blocks(content)
                if code_blocks:
                    ext_map = {"python": "py", "py": "py", "javascript": "js", "js": "js", "typescript": "ts", "ts": "ts", "html": "html", "css": "css", "json": "json", "sql": "sql"}
                    for c_idx, (c_lang, c_code) in enumerate(code_blocks):
                        ext = ext_map.get(c_lang.lower(), "txt")
                        st.download_button(
                            label=f"📥 Download Code Block {c_idx+1} (.{ext})",
                            data=c_code,
                            file_name=f"mindfulai_code_{msg_idx+1}_{c_idx+1}.{ext}",
                            mime="text/plain",
                            key=f"dl_code_{msg_idx}_{c_idx}",
                        )

    # Auto-TTS on fresh assistant response
    if "_auto_tts_text" in st.session_state:
        st.components.v1.html(TTSService.generate_browser_tts_js(st.session_state.pop("_auto_tts_text")), height=0, width=0)

    # Pending prompt processor
    if "_pending_prompt" in st.session_state:
        p_text = st.session_state.pop("_pending_prompt")
        execute_turn(prompt=p_text)

    # Bottom Multimodal Input Dock
    st.markdown("---")
    dock_c1, dock_c2 = st.columns([1, 2.5])

    with dock_c1:
        st.markdown("##### 🎤 Neural Voice Input")
        audio_record = mic_recorder(
            start_prompt="🔴 Start Recording",
            stop_prompt="⏹️ Transcribe & Send",
            just_once=True,
            key="chat_mic",
        )
        if audio_record and audio_record.get("bytes"):
            a_id = audio_record.get("id", str(hash(audio_record["bytes"])))
            if st.session_state.get("_last_mic_id") != a_id:
                st.session_state["_last_mic_id"] = a_id
                st.audio(audio_record["bytes"], format="audio/wav")
                execute_turn(
                    prompt="Please listen to this voice prompt and respond thoughtfully.",
                    audio_bytes=audio_record["bytes"],
                )

    with dock_c2:
        st.markdown("##### 📎 Multimodal Context & Studio Shortcuts")
        up_c1, up_c2, up_c3 = st.columns(3)
        with up_c1:
            up_doc = st.file_uploader("Document/Code", type=["pdf", "txt", "py", "js", "ts", "html", "css", "json", "md", "csv", "sql"], key="chat_up_doc")
        with up_c2:
            up_img = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"], key="chat_up_img")
        with up_c3:
            up_aud = st.file_uploader("Audio", type=["wav", "mp3", "m4a", "ogg"], key="chat_up_aud")

    _active_img = None
    _active_audio_bytes = None
    _active_doc_text = None
    _active_doc_name = None

    if up_doc:
        _raw = up_doc.getvalue()
        _active_doc_name = up_doc.name
        _active_doc_text = extract_file_content(_raw, up_doc.name)
        st.info(f"📄 **{up_doc.name}** loaded ({len(_raw):,} bytes). Ready for synthesis.")

    elif up_img:
        try:
            _active_img = Image.open(io.BytesIO(up_img.getvalue()))
            st.image(_active_img, width=160)
            st.info(f"🖼️ **{up_img.name}** attached for visual analysis or editing.")
        except Exception as e:
            st.error(f"Image load error: {e}")

    elif up_aud:
        _active_audio_bytes = up_aud.getvalue()
        st.audio(up_aud)
        st.info(f"🎙️ **{up_aud.name}** attached.")

    user_text = st.chat_input("Ask JARVIS anything, generate images ('/image lion'), or upload files...")
    if user_text:
        execute_turn(
            prompt=user_text,
            img_obj=_active_img,
            audio_bytes=_active_audio_bytes,
            doc_text=_active_doc_text,
            doc_name=_active_doc_name,
        )


# ===========================================================================
# MODE 2: 🎨 AI IMAGE STUDIO (TEXT-TO-IMAGE CREATIVE PLATFORM)
# ===========================================================================
elif st.session_state.studio_mode == "🎨 AI Image Studio":
    st.markdown("### 🎨 MindfulAI Image Studio — Text-to-Image Creation")
    st.markdown("Generate photorealistic, artistic, or 3D visuals using Google Imagen 3 with local ComfyUI fallback.")

    col_ctrl, col_prev = st.columns([1.1, 1.4])

    with col_ctrl:
        st.markdown('<div class="glass-card-3d">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎛️ Synthesis Parameters</div>', unsafe_allow_html=True)

        prompt_input = st.text_area(
            "Natural Language Prompt",
            placeholder="A realistic lion standing on a mountain peak at golden hour, hyper-detailed...",
            height=120,
        )

        st.caption("✨ Quick Style Inspirations:")
        insp_cols = st.columns(3)
        with insp_cols[0]:
            if st.button("🦁 Majestic Lion", key="insp_lion"):
                prompt_input = "A majestic lion sculpted from glowing neon glass and stardust in a cosmic nebula"
        with insp_cols[1]:
            if st.button("🤖 3D Robot", key="insp_robot"):
                prompt_input = "Futuristic 3D humanoid robot with glowing blue arc reactor and carbon fiber chassis"
        with insp_cols[2]:
            if st.button("🌆 Cyber City", key="insp_city"):
                prompt_input = "Breathtaking cyberpunk city skyline at night with flying cars and holographic neon billboards"

        aspect_choice = st.selectbox("Aspect Ratio", options=ASPECT_RATIOS, index=0)
        style_choice = st.selectbox("Aesthetic Style Preset", options=list(STYLE_PRESETS.keys()), index=1)
        engine_choice = st.selectbox("Rendering Engine", options=["Auto (Gemini Cloud → ComfyUI Local)", "Gemini Cloud (Imagen 3)", "ComfyUI (Local SD)"])

        neg_prompt = st.text_input("Negative Prompt (Optional)", value="blurry, low quality, artifacts, distorted, bad anatomy")

        seed_col1, seed_col2 = st.columns([0.7, 0.3])
        with seed_col1:
            random_seed = st.checkbox("Randomize Seed", value=True)
        with seed_col2:
            seed_val = None if random_seed else st.number_input("Seed", value=42, min_value=1)

        generate_btn = st.button("🚀 Render Visual", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_prev:
        st.markdown('<div class="glass-card-3d">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🖼️ Studio Canvas & Preview</div>', unsafe_allow_html=True)

        engine_target = "auto"
        if "ComfyUI" in engine_choice:
            engine_target = "comfyui"
        elif "Gemini" in engine_choice:
            engine_target = "gemini"

        if generate_btn and prompt_input.strip():
            with st.spinner("Synthesizing imagery..."):
                gen_result = ai_router.generation_service.generate(
                    prompt=prompt_input.strip(),
                    aspect_ratio=aspect_choice,
                    style=style_choice,
                    negative_prompt=neg_prompt,
                    seed=seed_val,
                    preferred_engine=engine_target,
                )

            if gen_result.success and gen_result.image_bytes:
                st.session_state.studio_gen_history.insert(0, {
                    "bytes": gen_result.image_bytes,
                    "prompt": prompt_input,
                    "provider": gen_result.provider,
                    "model": gen_result.model_used,
                    "aspect_ratio": aspect_choice,
                    "style": style_choice,
                    "time": time.strftime("%H:%M:%S"),
                })
                st.success(gen_result.message)
            else:
                st.error(gen_result.message)

        # Show latest generated image
        if st.session_state.studio_gen_history:
            latest = st.session_state.studio_gen_history[0]
            st.image(latest["bytes"], caption=f"Prompt: {latest['prompt']}", use_container_width=True)

            act_col1, act_col2, act_col3 = st.columns(3)
            with act_col1:
                st.markdown(f'<span class="prov-pill prov-gemini">Engine: {latest["provider"].upper()} · {latest["model"]}</span>', unsafe_allow_html=True)
            with act_col2:
                st.download_button(
                    label="📥 Download PNG",
                    data=latest["bytes"],
                    file_name="mindfulai_render.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with act_col3:
                if st.button("🪄 Open in Editor", use_container_width=True):
                    st.session_state.editor_source_bytes = latest["bytes"]
                    st.session_state.studio_mode = "🪄 Image Editor"
                    st.rerun()
        else:
            st.caption("No images generated yet. Configure parameters and click 'Render Visual' to synthesize your first creation.")

        st.markdown('</div>', unsafe_allow_html=True)

    # Recent Image Gallery
    st.markdown("#### 🖼️ Image Generation Gallery")
    saved_gens = image_storage_service.list_generated_images(limit=12)
    if saved_gens:
        g_cols = st.columns(4)
        for idx, g_item in enumerate(saved_gens):
            with g_cols[idx % 4]:
                f_path = Path(g_item.get("file_path", ""))
                if f_path.exists():
                    st.image(str(f_path), caption=g_item.get("prompt", "")[:35], use_container_width=True)
                    if st.button("🪄 Edit", key=f"edit_saved_{idx}", use_container_width=True):
                        st.session_state.editor_source_bytes = f_path.read_bytes()
                        st.session_state.studio_mode = "🪄 Image Editor"
                        st.rerun()


# ===========================================================================
# MODE 3: 🪄 DEDICATED IMAGE EDITOR (3-PANEL WORKSPACE)
# ===========================================================================
elif st.session_state.studio_mode == "🪄 Image Editor":
    st.markdown("### 🪄 MindfulAI Dedicated Image Editor Workspace")
    st.markdown("Professional 3-panel workspace: Upload any reference image and transform it through natural language commands.")

    p_left, p_center, p_right = st.columns([1, 1.4, 1.1])

    # ── LEFT PANEL: Source Upload & Metadata ──
    with p_left:
        st.markdown('<div class="glass-card-3d">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📤 1. Source Image</div>', unsafe_allow_html=True)

        uploaded_img = st.file_uploader("Upload Image to Edit", type=["png", "jpg", "jpeg", "webp"], key="dedicated_editor_upload")
        if uploaded_img:
            st.session_state.editor_source_bytes = uploaded_img.getvalue()

        if st.session_state.editor_source_bytes:
            src_bytes = st.session_state.editor_source_bytes
            st.image(src_bytes, caption="Active Reference Source", use_container_width=True)

            try:
                pil_src = Image.open(io.BytesIO(src_bytes))
                st.caption(f"📐 Resolution: `{pil_src.width} × {pil_src.height}` px")
                st.caption(f"📦 File Size: `{len(src_bytes) / 1024:.1f}` KB")
                st.caption(f"🎨 Color Mode: `{pil_src.mode}`")
            except Exception:
                pass

            if st.button("🗑️ Clear Image", use_container_width=True):
                st.session_state.editor_source_bytes = None
                st.session_state.editor_edited_bytes = None
                st.rerun()
        else:
            st.info("Upload an image here or generate one in Image Studio and click 'Open in Editor'.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── CENTER PANEL: Large Canvas & Before/After Comparison ──
    with p_center:
        st.markdown('<div class="glass-card-3d">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 2. Canvas & Comparison</div>', unsafe_allow_html=True)

        if st.session_state.editor_edited_bytes and st.session_state.editor_source_bytes:
            cmp_1, cmp_2 = st.columns(2)
            with cmp_1:
                st.markdown("**Original Reference**")
                st.image(st.session_state.editor_source_bytes, use_container_width=True)
            with cmp_2:
                st.markdown("**Transformed Result**")
                st.image(st.session_state.editor_edited_bytes, use_container_width=True)

            st.markdown(f"**Last Applied Command:** *{st.session_state.editor_instruction}*")
        elif st.session_state.editor_source_bytes:
            st.image(st.session_state.editor_source_bytes, caption="Awaiting Transformation Directive...", use_container_width=True)
        else:
            st.caption("Upload an image on the left to activate the central canvas.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT PANEL: Prompt, Presets & Actions ──
    with p_right:
        st.markdown('<div class="glass-card-3d">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🪄 3. Edit Directives</div>', unsafe_allow_html=True)

        preset_choice = st.selectbox(
            "Quick Tool Presets",
            options=["Custom Directive"] + list(EDIT_PRESETS.keys()),
            index=0,
        )

        initial_inst = ""
        if preset_choice != "Custom Directive":
            initial_inst = EDIT_PRESETS[preset_choice]

        edit_prompt = st.text_area(
            "Transformation Prompt",
            value=initial_inst,
            placeholder="e.g. Remove the background, replace with a beach, add sunglasses to the person, make it cinematic...",
            height=110,
        )

        apply_btn = st.button("✨ Apply Edit", use_container_width=True, type="primary")

        if apply_btn:
            if not st.session_state.editor_source_bytes:
                st.warning("Please upload a source image first.")
            elif not edit_prompt.strip():
                st.warning("Please enter an editing prompt or pick a preset.")
            else:
                with st.spinner("Executing reference-guided visual transformation..."):
                    res = ai_router.editing_service.edit(
                        image_bytes=st.session_state.editor_source_bytes,
                        instruction=edit_prompt.strip(),
                        preset_name=preset_choice if preset_choice != "Custom Directive" else None,
                    )

                if res.success and res.edited_bytes:
                    st.session_state.editor_edited_bytes = res.edited_bytes
                    st.session_state.editor_instruction = edit_prompt.strip()
                    st.success(res.message)
                    st.rerun()
                else:
                    st.error(res.message)

        if st.session_state.editor_edited_bytes:
            st.divider()
            st.markdown("##### 🔁 Iterative Actions")

            # Continue Editing: sets the edited image as the new source!
            if st.button("🔁 Continue Editing (Use as New Source)", use_container_width=True):
                st.session_state.editor_source_bytes = st.session_state.editor_edited_bytes
                st.session_state.editor_edited_bytes = None
                st.session_state.editor_instruction = ""
                st.success("Edited image set as new reference source! Enter your next instruction.")
                st.rerun()

            st.download_button(
                label="📥 Download Edited PNG",
                data=st.session_state.editor_edited_bytes,
                file_name="mindfulai_edited.png",
                mime="image/png",
                use_container_width=True,
            )

            if st.button("🔄 Reset to Original", use_container_width=True):
                st.session_state.editor_edited_bytes = None
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# MODE 4: 🎙️ VOICE ASSISTANT (AUDIO WORKSPACE)
# ===========================================================================
elif st.session_state.studio_mode == "🎙️ Voice Assistant":
    st.markdown("### 🎙️ MindfulAI Voice Assistant & Audio Matrix")
    st.markdown("Speech transcription powered by Gemini Audio Input and neural browser voice playback.")

    v_c1, v_c2 = st.columns(2)

    with v_c1:
        st.markdown('<div class="glass-card-3d">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔴 Voice Recording & Transcription</div>', unsafe_allow_html=True)
        st.caption("Record live speech using your microphone:")

        v_data = mic_recorder(
            start_prompt="🔴 Start Recording Voice",
            stop_prompt="⏹️ Complete & Transcribe",
            just_once=True,
            key="voice_studio_mic",
        )

        if v_data and v_data.get("bytes"):
            st.audio(v_data["bytes"], format="audio/wav")
            if st.button("⚡ Transcribe Waveform", key="btn_transcribe_voice_tab"):
                with st.spinner("Transcribing speech..."):
                    txt, prov, model = ai_router.audio_service.process_audio(
                        audio_bytes=v_data["bytes"],
                        user_prompt="Transcribe this speech accurately with punctuation.",
                    )
                st.markdown(f'<span class="prov-pill prov-gemini">Engine: {prov.upper()} · {model}</span>', unsafe_allow_html=True)
                st.markdown(f"**Transcription:**\n\n{txt}")

        st.markdown('</div>', unsafe_allow_html=True)

    with v_c2:
        st.markdown('<div class="glass-card-3d">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔊 Neural Speech Synthesizer (TTS)</div>', unsafe_allow_html=True)
        tts_input = st.text_area(
            "Speech Synthesis Text",
            value="Hello! I am MindfulAI, your JARVIS-style multimodal assistant. How can I assist your mission today?",
            height=130,
        )
        if st.button("📢 Synthesize Speech", key="btn_voice_tab_tts", use_container_width=True, type="primary"):
            st.components.v1.html(TTSService.generate_browser_tts_js(tts_input), height=0, width=0)
            st.success("Spoken aloud via neural speech engine.")
        st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# MODE 5: ⚙️ ENGINE MATRIX & DIAGNOSTICS
# ===========================================================================
elif st.session_state.studio_mode == "⚙️ Engine Matrix":
    st.markdown("### ⚙️ MindfulAI Engine Matrix & Telemetry")
    st.markdown("Real-time telemetry and health monitoring for all AI, image, and database engines.")

    e_1, e_2, e_3 = st.columns(3)

    with e_1:
        st.markdown('<div class="glass-card-3d">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">⚡ Google Gemini Cloud</div>', unsafe_allow_html=True)
        st.markdown(f"**API Key:** `{masked_key}`")
        st.markdown(f"**Text & Reasoning:** `{config.TEXT_MODEL}`")
        st.markdown(f"**Image Generation:** `{config.IMAGEN_MODEL}`")
        st.markdown(f"**Audio Input Model:** `{config.AUDIO_INPUT_MODEL}`")
        st.markdown(f"**TTS Speech Model:** `{config.TTS_MODEL}`")
        st.markdown(f"**Quota Status:** `{'PAUSED (Daily 429)' if is_daily_quota else 'HEALTHY (Active)'}`")

        if is_daily_quota and st.button("Unlock Quota Flag", key="btn_clear_diag_tab"):
            quota_manager.clear_daily_quota()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with e_2:
        st.markdown('<div class="glass-card-3d">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🖥️ Ollama Local Engine</div>', unsafe_allow_html=True)
        st.markdown(f"**Status:** `{'ONLINE' if ollama_running else 'STANDBY (Offline)'}`")
        st.markdown(f"**Base URL:** `{config.OLLAMA_BASE_URL}`")
        st.markdown(f"**Text Model:** `{config.OLLAMA_TEXT_MODEL}`")
        st.markdown(f"**Vision Model:** `{config.OLLAMA_VISION_MODEL}`")
        st.markdown(f"**Automatic Failover:** `{config.OLLAMA_FALLBACK_ON_DAILY_QUOTA}`")

        if st.button("Ping Ollama", key="btn_ping_ollama_tab"):
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with e_3:
        st.markdown('<div class="glass-card-3d">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎨 ComfyUI Local Studio</div>', unsafe_allow_html=True)
        st.markdown(f"**Status:** `{'ONLINE' if comfy_running else 'STANDBY (Offline)'}`")
        st.markdown(f"**Base URL:** `{config.COMFYUI_BASE_URL}`")
        st.markdown(f"**Target Checkpoint:** `{config.COMFYUI_MODEL}`")

        if comfy_running:
            stats = comfy_svc.get_system_stats()
            devices = stats.get("devices", [])
            if devices:
                st.caption(f"Hardware: {devices[0].get('name', 'GPU')} ({devices[0].get('vram_total', 0)/(1024**3):.1f} GB VRAM)")
        else:
            st.caption("Launch ComfyUI on default port 8188 to activate local diffusion rendering.")

        if st.button("Ping ComfyUI", key="btn_ping_comfy_tab"):
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
