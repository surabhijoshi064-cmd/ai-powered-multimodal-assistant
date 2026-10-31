"""
Multimodal chat endpoint — accepts image, audio, and document file uploads
alongside a text message via multipart/form-data.
"""
import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from typing import Optional
from app.models.database import get_db
from app.schemas.chat import ChatResponse
from app.services.chat_service import ChatService
from app.services.multimodal_service import MultimodalService
from app.config import settings
from app.utils.logging_config import setup_logging

logger = setup_logging(settings.LOG_LEVEL)

router = APIRouter(prefix="/api/v1/multimodal", tags=["Multimodal"])

chat_service = ChatService()
multimodal_service = MultimodalService()


@router.post("/chat", status_code=status.HTTP_200_OK)
def multimodal_chat(
    db: Session = Depends(get_db),
    message: str = Form(""),
    conversation_id: Optional[str] = Form(None),
    input_modality: str = Form("text"),
    file: Optional[UploadFile] = File(None),
):
    """
    Process a multimodal chat message. Accepts optional file (image, audio, document)
    alongside a text prompt. Returns a ChatResponse.
    """
    try:
        # Determine what kind of file was uploaded
        file_description = ""
        image_bytes = None
        transcribed_text = ""

        if file and file.filename:
            content = file.file.read()
            file.file.seek(0)
            ext = os.path.splitext(file.filename)[1].lower().lstrip(".")
            file_size = len(content)

            if file_size == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is empty.",
                )
            if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
                max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds {max_mb:.0f} MB limit.",
                )

            # ── Image upload → Gemini vision analysis ──
            if ext in settings.allowed_img_exts:
                input_modality = "image"
                image_bytes = content
                vision_description = multimodal_service.analyze_image(
                    image_bytes=content,
                    filename=file.filename,
                    user_prompt=message or "Describe this image in detail.",
                )
                file_description = f"[Image Analysis of {file.filename}]:\n{vision_description}"

            # ── Audio upload → speech-to-text transcription ──
            elif ext in settings.allowed_aud_exts:
                input_modality = "audio"
                safe_name = f"{uuid.uuid4().hex}.{ext}"
                audio_path = os.path.join(settings.AUDIO_DIR, safe_name)
                with open(audio_path, "wb") as f:
                    f.write(content)
                transcribed_text = multimodal_service.transcribe_audio(audio_path)
                file_description = f"[Audio Transcription of {file.filename}]:\n{transcribed_text}"

            # ── Document upload → text extraction for inline context ──
            elif ext in settings.allowed_doc_exts:
                input_modality = "document"
                doc_text = multimodal_service.extract_document_text(content, ext)
                file_description = (
                    f"[Document Content from {file.filename}]:\n"
                    f"<uploaded_document>\n{doc_text[:8000]}\n</uploaded_document>"
                )
            else:
                # Code / text files — read as text
                try:
                    text_content = content.decode("utf-8")
                except UnicodeDecodeError:
                    text_content = content.decode("latin-1", errors="replace")
                input_modality = "document"
                file_description = (
                    f"[File Content from {file.filename}]:\n"
                    f"```\n{text_content[:8000]}\n```"
                )

        # Build final message to send to chat pipeline
        final_message = message.strip()
        if file_description:
            if final_message:
                final_message = f"{final_message}\n\n{file_description}"
            else:
                final_message = file_description

        if not final_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A message or file upload is required.",
            )

        # Use the existing ChatService pipeline via a synthetic ChatRequest
        from app.schemas.chat import ChatRequest
        chat_request = ChatRequest(
            conversation_id=conversation_id,
            message=final_message,
            input_modality=input_modality,
        )

        response = chat_service.process_chat(db, chat_request)
        return response

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error in multimodal chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process multimodal message.",
        )

