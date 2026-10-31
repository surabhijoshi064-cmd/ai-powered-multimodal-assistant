import os
from typing import List, Optional
from sqlalchemy.orm import Session
from app.models.conversation import Conversation
from app.models.document import Document
from app.config import settings
from app.utils.logging_config import setup_logging

logger = setup_logging(settings.LOG_LEVEL)


class ConversationService:
    @staticmethod
    def create_conversation(db: Session, title: str = "New Conversation") -> Conversation:
        conversation = Conversation(title=title)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        logger.info(f"Created conversation {conversation.id} titled '{title}'")
        return conversation

    @staticmethod
    def list_conversations(db: Session, limit: int = 50, offset: int = 0) -> List[Conversation]:
        return (
            db.query(Conversation)
            .order_by(Conversation.updated_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    @staticmethod
    def count_conversations(db: Session) -> int:
        return db.query(Conversation).count()

    @staticmethod
    def get_conversation(db: Session, conversation_id: str) -> Optional[Conversation]:
        return db.query(Conversation).filter(Conversation.id == conversation_id).first()

    @staticmethod
    def delete_conversation(db: Session, conversation_id: str) -> bool:
        conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
        if not conv:
            return False

        # Clean up any document files stored on disk for this conversation
        docs = db.query(Document).filter(Document.conversation_id == conversation_id).all()
        for doc in docs:
            safe_path = os.path.join(settings.UPLOAD_DIR, doc.safe_filename)
            if os.path.exists(safe_path):
                try:
                    os.remove(safe_path)
                    logger.info(f"Cleaned up file {safe_path} on conversation deletion")
                except OSError as e:
                    logger.warning(f"Failed to remove file {safe_path}: {e}")

        db.delete(conv)
        db.commit()
        logger.info(f"Deleted conversation {conversation_id} and associated records")
        return True
