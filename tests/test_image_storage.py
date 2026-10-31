"""
Unit tests for ImageStorageService.
"""
import io
from pathlib import Path
from PIL import Image
from services.image_storage_service import image_storage_service


def test_save_and_list_generated_image(tmp_path):
    img = Image.new("RGB", (32, 32), color="purple")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    entry = image_storage_service.save_generated_image(
        image_bytes=raw_bytes,
        prompt="A glowing purple cube",
        provider="gemini",
        model="imagen-3.0-generate-002",
        aspect_ratio="1:1",
        style="3D Render",
    )

    assert entry["id"].startswith("gen_")
    assert Path(entry["file_path"]).exists()

    gens = image_storage_service.list_generated_images(limit=5)
    assert len(gens) > 0
    assert any(g["id"] == entry["id"] for g in gens)


def test_save_and_list_edited_image(tmp_path):
    orig = Image.new("RGB", (32, 32), color="red")
    edit = Image.new("RGB", (32, 32), color="blue")

    b_orig, b_edit = io.BytesIO(), io.BytesIO()
    orig.save(b_orig, format="PNG")
    edit.save(b_edit, format="PNG")

    entry = image_storage_service.save_edited_image(
        edited_bytes=b_edit.getvalue(),
        original_bytes=b_orig.getvalue(),
        instruction="Change color to blue",
        provider="gemini",
        model="imagen-3.0-generate-002",
        preset="Color Grading",
    )

    assert entry["id"].startswith("edit_")
    assert Path(entry["original_path"]).exists()
    assert Path(entry["edited_path"]).exists()

    edits = image_storage_service.list_edited_images(limit=5)
    assert len(edits) > 0
    assert any(e["id"] == entry["id"] for e in edits)

