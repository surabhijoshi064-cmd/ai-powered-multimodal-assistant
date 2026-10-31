import os
import re
import uuid
from typing import Tuple
from fastapi import UploadFile, HTTPException, status
from app.config import settings

# PDF magic bytes: '%PDF-'
PDF_MAGIC_BYTES = b"%PDF-"


class FileValidator:
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Strip directory components and unsafe characters from original filename."""
        base = os.path.basename(filename)
        # Remove any characters other than alphanumeric, dots, underscores, dashes
        clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", base)
        return clean or "unnamed_file"

    @classmethod
    def generate_safe_filename(cls, original_filename: str) -> str:
        """Generate a collision-resistant UUID filename while preserving safe extension."""
        sanitized = cls.sanitize_filename(original_filename)
        ext = os.path.splitext(sanitized)[1].lower()
        return f"{uuid.uuid4().hex}{ext}"

    @classmethod
    async def validate_document_file(cls, file: UploadFile) -> Tuple[bytes, str, str]:
        """
        Validates document file type, size, and magic numbers.
        Returns (content_bytes, original_filename, safe_filename).
        """
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must have a filename",
            )

        original_filename = cls.sanitize_filename(file.filename)
        ext = os.path.splitext(original_filename)[1].lower().lstrip(".")

        if ext not in settings.allowed_doc_exts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(settings.allowed_doc_exts)}",
            )

        # Read content and enforce size limits
        content = await file.read()
        file_size = len(content)

        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)",
            )

        if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
            max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed limit of {max_mb:.1f} MB",
            )

        # Content verification / Magic numbers check
        if ext == "pdf":
            if not content.startswith(PDF_MAGIC_BYTES):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid PDF file format (missing PDF magic header)",
                )
        else:
            # Verify text / code file is decodeable
            try:
                content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    content.decode("latin-1")
                except Exception:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid text file: content cannot be decoded as text",
                    )

        safe_filename = cls.generate_safe_filename(original_filename)
        return content, original_filename, safe_filename
