"""
File Validation Utilities for MindfulAI.

Validates document, image, and audio uploads against allowed extensions and size limits.
Prevents processing malicious or oversize files.
"""
import os
from typing import Tuple, Optional
import config


class FileValidator:
    """Enforces file upload constraints."""

    @staticmethod
    def validate_document(filename: str, file_size_bytes: int) -> Tuple[bool, Optional[str]]:
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext not in config.ALLOWED_DOCUMENT_EXTENSIONS:
            return False, f"File extension '.{ext}' is not supported. Allowed: {', '.join(config.ALLOWED_DOCUMENT_EXTENSIONS)}."
        if file_size_bytes > config.MAX_UPLOAD_SIZE_BYTES:
            return False, f"File '{filename}' exceeds {config.MAX_UPLOAD_SIZE_BYTES / (1024*1024):.0f} MB limit."
        return True, None

    @staticmethod
    def validate_image(filename: str, file_size_bytes: int) -> Tuple[bool, Optional[str]]:
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext not in config.ALLOWED_IMAGE_EXTENSIONS:
            return False, f"Image extension '.{ext}' is not supported. Allowed: {', '.join(config.ALLOWED_IMAGE_EXTENSIONS)}."
        if file_size_bytes > config.MAX_UPLOAD_SIZE_BYTES:
            return False, f"Image exceeds {config.MAX_UPLOAD_SIZE_BYTES / (1024*1024):.0f} MB limit."
        return True, None

    @staticmethod
    def validate_audio(filename: str, file_size_bytes: int) -> Tuple[bool, Optional[str]]:
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext not in config.ALLOWED_AUDIO_EXTENSIONS:
            return False, f"Audio format '.{ext}' is not supported. Allowed: {', '.join(config.ALLOWED_AUDIO_EXTENSIONS)}."
        if file_size_bytes > config.MAX_UPLOAD_SIZE_BYTES:
            return False, f"Audio exceeds {config.MAX_UPLOAD_SIZE_BYTES / (1024*1024):.0f} MB limit."
        return True, None

