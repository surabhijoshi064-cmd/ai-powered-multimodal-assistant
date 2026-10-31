"""
Streamlit Session State Manager.

Coordinates multi-turn conversations, active conversation IDs, and audio recording states.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Any
import streamlit as st


class SessionManager:
    """Encapsulates Streamlit session state initialization and mutation."""

    @staticmethod
    def initialize():
        if "conversations" not in st.session_state:
            initial_id = str(uuid.uuid4())
            st.session_state.conversations = {
                initial_id: {
                    "title": "New Conversation",
                    "messages": [],
                    "created_at": datetime.now(),
                }
            }
            st.session_state.current_conv_id = initial_id

        if "current_conv_id" not in st.session_state:
            st.session_state.current_conv_id = list(st.session_state.conversations.keys())[0]

        if "tts_enabled" not in st.session_state:
            st.session_state.tts_enabled = False

        if "_last_mic_id" not in st.session_state:
            st.session_state._last_mic_id = None

        if "_request_in_flight" not in st.session_state:
            st.session_state["_request_in_flight"] = False

    @staticmethod
    def get_current_messages() -> List[Dict[str, Any]]:
        cid = st.session_state.get("current_conv_id")
        if cid and cid in st.session_state.get("conversations", {}):
            return st.session_state.conversations[cid]["messages"]
        return []

    @staticmethod
    def append_message(role: str, content: str, **kwargs):
        cid = st.session_state.get("current_conv_id")
        if cid and cid in st.session_state.get("conversations", {}):
            msg = {"role": role, "content": content, **kwargs}
            st.session_state.conversations[cid]["messages"].append(msg)

    @staticmethod
    def create_conversation() -> str:
        new_id = str(uuid.uuid4())
        st.session_state.conversations[new_id] = {
            "title": "New Conversation",
            "messages": [],
            "created_at": datetime.now(),
        }
        st.session_state.current_conv_id = new_id
        return new_id

    @staticmethod
    def delete_conversation(conv_id: str):
        if conv_id in st.session_state.conversations:
            del st.session_state.conversations[conv_id]
            if st.session_state.current_conv_id == conv_id:
                remaining = list(st.session_state.conversations.keys())
                if remaining:
                    st.session_state.current_conv_id = remaining[0]
                else:
                    SessionManager.create_conversation()

