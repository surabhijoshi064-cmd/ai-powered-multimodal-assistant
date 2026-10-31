"""
Image Storage and History Service for MindfulAI.

Manages persistent on-disk storage of generated and edited images in:
  - storage/generated_images/
  - storage/edited_images/
Provides indexing, metadata tracking, and retrieval for the Studio Gallery.
"""
import io
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

import config

logger = logging.getLogger(__name__)

STORAGE_ROOT = Path(__file__).resolve().parent.parent / "storage"
GENERATED_DIR = STORAGE_ROOT / "generated_images"
EDITED_DIR = STORAGE_ROOT / "edited_images"
INDEX_FILE = STORAGE_ROOT / "images_index.json"

for d in [GENERATED_DIR, EDITED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class ImageStorageService:
    """Manages persistent image files and metadata index."""

    def __init__(self):
        self._ensure_index()

    def _ensure_index(self):
        if not INDEX_FILE.exists():
            try:
                INDEX_FILE.write_text(json.dumps({"generated": [], "edited": []}, indent=2), encoding="utf-8")
            except Exception as e:
                logger.debug(f"Could not create image index file: {e}")

    def _load_index(self) -> Dict[str, List[Dict[str, Any]]]:
        self._ensure_index()
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"generated": [], "edited": []}

    def _save_index(self, data: Dict[str, List[Dict[str, Any]]]):
        try:
            with open(INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save image index: {e}")

    def save_generated_image(
        self,
        image_bytes: bytes,
        prompt: str,
        provider: str,
        model: str,
        aspect_ratio: str = "1:1",
        style: str = "Cinematic",
    ) -> Dict[str, Any]:
        """Save a generated image to disk and index."""
        img_id = f"gen_{uuid.uuid4().hex[:10]}"
        filename = f"{img_id}.png"
        file_path = GENERATED_DIR / filename

        try:
            file_path.write_bytes(image_bytes)
        except Exception as e:
            logger.error(f"Failed to write generated image file {file_path}: {e}")

        entry = {
            "id": img_id,
            "filename": filename,
            "file_path": str(file_path),
            "prompt": prompt,
            "provider": provider,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "style": style,
            "timestamp": time.time(),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        index = self._load_index()
        index["generated"].insert(0, entry)
        if len(index["generated"]) > 100:
            index["generated"] = index["generated"][:100]
        self._save_index(index)

        return entry

    def save_edited_image(
        self,
        edited_bytes: bytes,
        original_bytes: bytes,
        instruction: str,
        provider: str,
        model: str,
        preset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save both original and edited images to disk and index."""
        edit_id = f"edit_{uuid.uuid4().hex[:10]}"
        orig_filename = f"{edit_id}_orig.png"
        edit_filename = f"{edit_id}_out.png"

        orig_path = EDITED_DIR / orig_filename
        edit_path = EDITED_DIR / edit_filename

        try:
            orig_path.write_bytes(original_bytes)
            edit_path.write_bytes(edited_bytes)
        except Exception as e:
            logger.error(f"Failed to write edited image files: {e}")

        entry = {
            "id": edit_id,
            "original_filename": orig_filename,
            "edited_filename": edit_filename,
            "original_path": str(orig_path),
            "edited_path": str(edit_path),
            "instruction": instruction,
            "provider": provider,
            "model": model,
            "preset": preset or "",
            "timestamp": time.time(),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        index = self._load_index()
        index["edited"].insert(0, entry)
        if len(index["edited"]) > 100:
            index["edited"] = index["edited"][:100]
        self._save_index(index)

        return entry

    def list_generated_images(self, limit: int = 24) -> List[Dict[str, Any]]:
        """List recently generated images."""
        index = self._load_index()
        return index.get("generated", [])[:limit]

    def list_edited_images(self, limit: int = 24) -> List[Dict[str, Any]]:
        """List recently edited images."""
        index = self._load_index()
        return index.get("edited", [])[:limit]


# Global singleton
image_storage_service = ImageStorageService()

