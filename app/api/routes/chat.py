from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.utils.logging_config import setup_logging
from app.config import settings

logger = setup_logging(settings.LOG_LEVEL)

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])

chat_service = ChatService()


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def send_chat_message(
    payload: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    Process a chat message with conversation history and generate an empathetic response.
    """
    try:
        response = chat_service.process_chat(db, payload)
        return response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat message",
        )
