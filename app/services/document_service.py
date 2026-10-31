import os
import pymupdf as fitz  # PyMuPDF
from typing import List, Optional, Tuple
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.orm import Session
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.security.file_validation import FileValidator
from app.retrieval.chunking import TextChunker
from app.providers.embedding_provider import get_embedding_provider
from app.retrieval.vector_store import vector_store
from app.config import settings
from app.utils.logging_config import setup_logging

logger = setup_logging(settings.LOG_LEVEL)


class DocumentService:
    def __init__(self):
        self.chunker = TextChunker()
        self.embedding_provider = get_embedding_provider()

    async def process_and_store_document(
        self,
        db: Session,
        file: UploadFile,
        conversation_id: Optional[str] = None,
    ) -> Document:
        # 1. Validate file format and size
        content, original_filename, safe_filename = await FileValidator.validate_document_file(file)

        # 2. Save physical file in upload storage directory
        file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)
        with open(file_path, "wb") as f:
            f.write(content)

        # 3. Create Document DB record
        doc_record = Document(
            conversation_id=conversation_id,
            original_filename=original_filename,
            safe_filename=safe_filename,
            mime_type=file.content_type or "application/octet-stream",
            file_size=len(content),
            processing_status="processing",
        )
        db.add(doc_record)
        db.commit()
        db.refresh(doc_record)

        # 4. Extract text based on file type
        pages: List[Tuple[Optional[int], str]] = []
        ext = os.path.splitext(original_filename)[1].lower().lstrip(".")

        try:
            if ext == "pdf":
                with fitz.open(stream=content, filetype="pdf") as pdf_doc:
                    for page_idx in range(len(pdf_doc)):
                        page_text = pdf_doc[page_idx].get_text()
                        pages.append((page_idx + 1, page_text))
            else:
                try:
                    text_content = content.decode("utf-8")
                except UnicodeDecodeError:
                    text_content = content.decode("latin-1")
                pages.append((1, text_content))

            # 5. Chunk extracted text
            chunks_data = self.chunker.chunk_pages(pages)
            if not chunks_data:
                raise ValueError("Document contains no readable text or is empty")

            # 6. Save chunks in DB
            db_chunks = []
            chunk_dicts_for_store = []
            texts_to_embed = []

            for c in chunks_data:
                db_chunk = DocumentChunk(
                    document_id=doc_record.id,
                    chunk_index=c["chunk_index"],
                    chunk_text=c["chunk_text"],
                    page_number=c["page_number"],
                )
                db.add(db_chunk)
                db_chunks.append(db_chunk)
                texts_to_embed.append(c["chunk_text"])

            db.commit()

            # Refresh all chunks to obtain IDs
            for ch in db_chunks:
                db.refresh(ch)
                chunk_dicts_for_store.append({
                    "id": ch.id,
                    "document_id": doc_record.id,
                    "document_name": doc_record.original_filename,
                    "page_number": ch.page_number,
                    "chunk_index": ch.chunk_index,
                    "chunk_text": ch.chunk_text,
                    "conversation_id": conversation_id,
                })

            # 7. Generate vector embeddings & add to vector store
            embeddings = self.embedding_provider.embed_texts(texts_to_embed)
            vector_store.add_chunks(chunk_dicts_for_store, embeddings)

            # 8. Mark document as completed
            doc_record.processing_status = "completed"
            db.commit()
            db.refresh(doc_record)
            logger.info(
                f"Successfully processed document '{original_filename}' ({len(db_chunks)} chunks)"
            )
            return doc_record

        except Exception as e:
            logger.error(f"Error processing document {original_filename}: {e}", exc_info=True)
            doc_record.processing_status = "failed"
            doc_record.error_message = str(e)
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to process document: {str(e)}",
            )

    @staticmethod
    def list_documents(db: Session, limit: int = 50, offset: int = 0) -> List[Document]:
        return (
            db.query(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    @staticmethod
    def count_documents(db: Session) -> int:
        return db.query(Document).count()

    @staticmethod
    def get_document(db: Session, document_id: str) -> Optional[Document]:
        return db.query(Document).filter(Document.id == document_id).first()

    @staticmethod
    def delete_document(db: Session, document_id: str) -> bool:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return False

        # Remove physical file
        safe_path = os.path.join(settings.UPLOAD_DIR, doc.safe_filename)
        if os.path.exists(safe_path):
            try:
                os.remove(safe_path)
            except OSError as e:
                logger.warning(f"Failed to delete document file {safe_path}: {e}")

        # Remove from vector store
        vector_store.delete_document_chunks(doc.id)

        # Remove from DB (chunks cascade automatically)
        db.delete(doc)
        db.commit()
        logger.info(f"Deleted document {document_id}")
        return True
