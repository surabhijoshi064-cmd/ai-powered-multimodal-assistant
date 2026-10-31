import json
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.chat import ChatRequest, ChatResponse, Citation
from app.providers.llm_provider import get_llm_provider, get_ollama_provider_or_none
from app.services.conversation_service import ConversationService
from app.services.rag_service import RAGService
from app.services.guardrail_service import GuardrailService
from app.services.emotional_support_service import EmotionalSupportService
from app.config import settings
from app.utils.logging_config import setup_logging
from app.utils.provider_state import (
    mark_gemini_daily_quota_exhausted,
    is_gemini_daily_quota_exhausted,
    get_quota_reset_eta_hours,
    set_active_provider,
)

logger = setup_logging(settings.LOG_LEVEL)

DEFAULT_SYSTEM_PROMPT = (
    "You are an empathetic, compassionate, and secure AI assistant. "
    "Provide nonjudgmental emotional support, thoughtful listening, and practical grounding strategies "
    "such as breathing exercises or structured reflection when helpful. "
    "Maintain clear boundaries: you are an AI support tool, not a medical or licensed mental health provider. "
    "Do not provide medical prescriptions, diagnoses, or clinical advice."
)


def _build_daily_quota_message(retry_hrs: int, ollama_available: bool, ollama_error: str = "") -> str:
    """Build a user-facing message for Gemini daily quota exhaustion."""
    base = (
        "⏳ **Gemini's API quota has been reached.**\n\n"
        "Your chatbot is running, but Google is temporarily restricting additional requests. "
        f"The daily free-tier limit has been exceeded. "
        f"Google will restore access in approximately **{retry_hrs} hour(s)**.\n\n"
    )
    if ollama_available:
        return base + "✅ *Switched to local Ollama AI as fallback.*"
    elif ollama_error:
        return base + (
            f"> ⚠️ Ollama fallback also unavailable: *{ollama_error[:120]}*\n\n"
            "> *To enable local fallback: install Ollama, set `OLLAMA_ENABLED=true` in `.env`, "
            "and run `ollama serve`.*"
        )
    else:
        return base + (
            "> *To enable a local fallback: install Ollama, set `OLLAMA_ENABLED=true` in `.env`, "
            "pull a model (`ollama pull gemma2:2b`), and run `ollama serve`.*\n\n"
            "> *This is a Google API restriction — no code change can bypass it. "
            "Please wait for the quota to reset or upgrade your Google AI Studio plan.*"
        )


class ChatService:
    def __init__(self, llm_provider=None):
        self._llm_provider = llm_provider
        self.rag_service = RAGService()

    @property
    def llm_provider(self):
        return self._llm_provider or get_llm_provider()

    def process_chat(
        self,
        db: Session,
        request: ChatRequest,
        rag_context: Optional[str] = None,
        citations: Optional[List[Citation]] = None,
    ) -> ChatResponse:
        # 1. Sanitize user input
        cleaned_message = GuardrailService.sanitize_user_input(request.message)

        # 2. Check for Prompt Injection / Rule Override attempts
        is_suspicious, injection_reason = GuardrailService.check_prompt_injection(cleaned_message)
        if is_suspicious:
            logger.warning(f"Blocked input due to guardrail alert: {injection_reason}")
            cleaned_message = f"[System Warning: Input sanitized] {cleaned_message}"

        # 3. Resolve or create conversation
        if request.conversation_id:
            conv = ConversationService.get_conversation(db, request.conversation_id)
            if not conv:
                raise ValueError(f"Conversation {request.conversation_id} not found")
        else:
            title = cleaned_message[:40] + ("..." if len(cleaned_message) > 40 else "")
            conv = ConversationService.create_conversation(db, title=title)

        # 4. Crisis Intervention Detection (High priority override)
        is_crisis, crisis_response = EmotionalSupportService.evaluate_crisis_risk(cleaned_message)
        if is_crisis:
            # Store user message
            user_msg = Message(
                conversation_id=conv.id,
                role="user",
                content=cleaned_message,
                input_modality=request.input_modality,
            )
            db.add(user_msg)
            
            # Store crisis response
            assistant_msg = Message(
                conversation_id=conv.id,
                role="assistant",
                content=crisis_response,
                input_modality="text",
                metadata_json=json.dumps({"crisis_detected": True}),
            )
            db.add(assistant_msg)
            db.commit()
            db.refresh(assistant_msg)

            return ChatResponse(
                conversation_id=conv.id,
                message_id=assistant_msg.id,
                role="assistant",
                response=crisis_response,
                input_modality="text",
                citations=None,
                is_grounded=False,
                crisis_detected=True,
            )

        # 5. Store user message in DB
        user_msg = Message(
            conversation_id=conv.id,
            role="user",
            content=cleaned_message,
            input_modality=request.input_modality,
        )
        db.add(user_msg)
        db.commit()

        # 6. Automatic RAG context retrieval if not already provided
        active_citations = citations or []
        active_rag_context = rag_context

        if not active_rag_context:
            retrieved_ctx, retrieved_citations = self.rag_service.retrieve_context(
                query=cleaned_message,
                conversation_id=conv.id,
            )
            if retrieved_ctx:
                active_rag_context = retrieved_ctx
                active_citations = retrieved_citations

        # 7. Retrieve conversation history (up to last 15 messages)
        history_msgs = (
            db.query(Message)
            .filter(Message.conversation_id == conv.id)
            .order_by(Message.created_at.asc())
            .limit(15)
            .all()
        )

        formatted_messages: List[Dict[str, str]] = []
        for m in history_msgs:
            formatted_messages.append({"role": m.role, "content": m.content})

        # 8. Integrate RAG context into prompt
        system_prompt = DEFAULT_SYSTEM_PROMPT
        is_grounded = bool(active_rag_context)

        if active_rag_context:
            system_prompt += (
                f"\n\n--- GROUNDED KNOWLEDGE BASE ---\n"
                f"{active_rag_context}\n"
                f"--- END KNOWLEDGE BASE ---\n\n"
                "Instructions: Answer using the retrieved knowledge when relevant. "
                "Explicitly reference source documents and page numbers where available. "
                "If the knowledge base does not contain sufficient evidence, state clearly that "
                "the uploaded documents do not contain that information, rather than fabricating."
            )

        # 9. Call LLM Provider
        try:
            assistant_text = self.llm_provider.generate_response(
                messages=formatted_messages,
                system_prompt=system_prompt,
            )
        except RuntimeError as e:
            err_str = str(e)
            logger.error(f"Error generating LLM response: {err_str}")

            if err_str.startswith("daily_quota::"):
                # Parse: "daily_quota::<seconds>::<message>"
                parts_err = err_str.split("::", 2)
                retry_secs = int(parts_err[1]) if len(parts_err) > 1 else 0
                retry_hrs = max(1, round(retry_secs / 3600))

                # Mark quota exhausted so future requests skip Gemini immediately
                mark_gemini_daily_quota_exhausted(retry_after_seconds=max(retry_secs, 3600))

                # --- Automatic Ollama fallback ---
                if settings.OLLAMA_ENABLED and settings.OLLAMA_FALLBACK_ON_DAILY_QUOTA:
                    logger.info(
                        "Gemini daily quota exhausted. Attempting automatic Ollama fallback."
                    )
                    ollama_provider = get_ollama_provider_or_none()
                    if ollama_provider is not None:
                        try:
                            assistant_text = ollama_provider.generate_response(
                                messages=formatted_messages,
                                system_prompt=system_prompt,
                            )
                            set_active_provider("ollama")
                            # Prepend a short notice so the user knows which provider answered
                            assistant_text = (
                                "*(ℹ️ Gemini daily quota reached — response provided by local Ollama AI)*\n\n"
                                + assistant_text
                            )
                        except RuntimeError as ollama_err:
                            ollama_err_str = str(ollama_err)
                            logger.warning(f"Ollama fallback also failed: {ollama_err_str}")
                            assistant_text = _build_daily_quota_message(
                                retry_hrs, ollama_available=False, ollama_error=ollama_err_str
                            )
                    else:
                        assistant_text = _build_daily_quota_message(retry_hrs, ollama_available=False)
                else:
                    assistant_text = _build_daily_quota_message(retry_hrs, ollama_available=False)

            elif err_str.startswith("rate_limit::"):
                # Parse: "rate_limit::<seconds>::<message>"
                parts_err = err_str.split("::", 2)
                retry_secs = int(float(parts_err[1])) if len(parts_err) > 1 else 60
                assistant_text = (
                    f"⏸️ **Gemini's per-minute request limit was reached.**\n\n"
                    f"Please wait approximately **{retry_secs} seconds** and try again. "
                    "The chatbot is pausing to respect Google's free-tier rate limits."
                )
            elif err_str.startswith("auth_failure::"):
                assistant_text = (
                    "🔑 **Gemini API Key Authentication Failed.**\n\n"
                    "Google rejected the provided API key. Please check your `GEMINI_API_KEY` in `.env` "
                    "or generate a new one at [Google AI Studio](https://aistudio.google.com/app/apikey)."
                )
            elif err_str.startswith("permission_denied::"):
                assistant_text = (
                    "🔒 **Gemini API Permission Denied (HTTP 403).**\n\n"
                    "Access is restricted by Google. Please ensure the Generative Language API is enabled "
                    "in your Google Cloud Console."
                )
            elif err_str.startswith("network_error::"):
                assistant_text = (
                    "🌐 **Network Connection Error.**\n\n"
                    "Unable to connect to Google's Gemini servers. Please check your internet connection "
                    "and try again."
                )
            elif err_str.startswith("model_not_found::"):
                assistant_text = (
                    "🔍 **Gemini Model Not Found.**\n\n"
                    "None of the candidate models were reachable for your account. "
                    "Please check your Google AI Studio project for accessible models."
                )
            # Ollama-specific errors
            elif err_str.startswith("ollama_unavailable::"):
                assistant_text = (
                    "🖥️ **Local Ollama Server Unavailable.**\n\n"
                    "Ollama is not running. Please start it with `ollama serve` in a terminal, "
                    "then try again."
                )
            elif err_str.startswith("ollama_model_not_found::"):
                model_name = settings.OLLAMA_MODEL
                assistant_text = (
                    f"📦 **Ollama Model Not Found: `{model_name}`**\n\n"
                    f"The model has not been downloaded yet. Run this in your terminal:\n"
                    f"```\nollama pull {model_name}\n```\n"
                    "Then try again."
                )
            elif err_str.startswith("ollama_timeout::"):
                assistant_text = (
                    "⏳ **Ollama Request Timed Out.**\n\n"
                    "The local model is taking longer than expected — it may be loading for the first time. "
                    "Please try again in a moment."
                )
            elif err_str.startswith("ollama_error::"):
                assistant_text = (
                    "⚠️ **Ollama Error.**\n\n"
                    "The local AI model returned an unexpected error. Check that Ollama is running "
                    "and the model is available."
                )
            else:
                assistant_text = (
                    "I apologize, but I am currently unable to generate a response. "
                    "Please verify your provider configuration in `.env` and try again."
                )
        except Exception as e:
            logger.error(f"Unexpected error generating LLM response: {e}")
            assistant_text = (
                "I apologize, but an unexpected error occurred. "
                "Please try again in a moment."
            )

        # 10. Store assistant message in DB
        metadata_dict = {
            "is_grounded": is_grounded,
            "citations": [c.model_dump() for c in active_citations] if active_citations else [],
            "crisis_detected": False,
        }

        assistant_msg = Message(
            conversation_id=conv.id,
            role="assistant",
            content=assistant_text,
            input_modality="text",
            metadata_json=json.dumps(metadata_dict),
        )
        db.add(assistant_msg)
        db.commit()
        db.refresh(assistant_msg)

        return ChatResponse(
            conversation_id=conv.id,
            message_id=assistant_msg.id,
            role="assistant",
            response=assistant_text,
            input_modality="text",
            citations=active_citations if active_citations else None,
            is_grounded=is_grounded,
            crisis_detected=False,
        )
