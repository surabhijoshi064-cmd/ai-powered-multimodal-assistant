from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


class DocumentChunkSchema(BaseModel):
    id: str
    chunk_index: int
    chunk_text: str
    page_number: Optional[int] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentResponse(BaseModel):
    id: str
    conversation_id: Optional[str] = None
    original_filename: str
    mime_type: str
    file_size: int
    processing_status: str
    error_message: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentDetailResponse(DocumentResponse):
    chunks: List[DocumentChunkSchema] = []

    model_config = ConfigDict(from_attributes=True)


class DocumentUploadResponse(BaseModel):
    document_id: str
    original_filename: str
    chunk_count: int
    status: str
    message: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
