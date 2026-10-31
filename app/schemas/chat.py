from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Citation(BaseModel):
    document_name: str
    page_number: Optional[int] = None
    chunk_index: int
    snippet: str
    score: Optional[float] = None


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = Field(
        None, description="Existing conversation UUID or null to create new"
    )
    message: str = Field(..., min_length=1, max_length=10000, description="User text message")
    input_modality: str = Field("text", description="Input type: text, image, audio, document")
    media_path: Optional[str] = Field(None, description="Optional path to attached media")


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    role: str = "assistant"
    response: str
    input_modality: str = "text"
    citations: Optional[List[Citation]] = None
    is_grounded: bool = False
    crisis_detected: bool = False
