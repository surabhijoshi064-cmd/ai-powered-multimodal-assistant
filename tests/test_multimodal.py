import io
from fastapi.testclient import TestClient


def test_multimodal_text_message(client: TestClient):
    """Verify multimodal endpoint works with a text-only prompt."""
    res = client.post(
        "/api/v1/multimodal/chat",
        data={"message": "Hello from multimodal test", "input_modality": "text"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "conversation_id" in data
    assert "response" in data
    assert len(data["response"]) > 0


def test_multimodal_file_upload(client: TestClient):
    """Verify multimodal endpoint accepts a document file attachment."""
    file_content = b"print('Hello world!')\n"
    files = {"file": ("test_script.py", io.BytesIO(file_content), "text/plain")}
    res = client.post(
        "/api/v1/multimodal/chat",
        data={"message": "Please refactor this code", "input_modality": "document"},
        files=files,
    )
    assert res.status_code == 200
    data = res.json()
    assert "conversation_id" in data
    assert len(data["response"]) > 0

