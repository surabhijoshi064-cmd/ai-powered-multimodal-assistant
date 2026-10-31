def test_conversation_lifecycle(client):
    # 1. Create a conversation
    create_res = client.post("/api/v1/conversations", json={"title": "Mental Wellness Checkin"})
    assert create_res.status_code == 201
    conv = create_res.json()
    conv_id = conv["id"]
    assert conv["title"] == "Mental Wellness Checkin"

    # 2. List conversations
    list_res = client.get("/api/v1/conversations")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1
    assert any(c["id"] == conv_id for c in list_data["conversations"])

    # 3. Send chat message to this conversation
    chat_payload = {
        "conversation_id": conv_id,
        "message": "Hello, I feel a bit overwhelmed today.",
        "input_modality": "text",
    }
    chat_res = client.post("/api/v1/chat", json=chat_payload)
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert chat_data["conversation_id"] == conv_id
    assert "response" in chat_data
    assert len(chat_data["response"]) > 0

    # 4. Send follow-up chat message
    followup_payload = {
        "conversation_id": conv_id,
        "message": "Can you give me a breathing technique?",
        "input_modality": "text",
    }
    followup_res = client.post("/api/v1/chat", json=followup_payload)
    assert followup_res.status_code == 200

    # 5. Fetch conversation detail with history
    detail_res = client.get(f"/api/v1/conversations/{conv_id}")
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert len(detail_data["messages"]) == 4  # 2 user messages + 2 assistant messages

    # 6. Delete conversation
    del_res = client.delete(f"/api/v1/conversations/{conv_id}")
    assert del_res.status_code == 204

    # 7. Verify conversation is gone
    get_res = client.get(f"/api/v1/conversations/{conv_id}")
    assert get_res.status_code == 404


def test_chat_auto_creates_conversation_when_none_provided(client):
    payload = {
        "message": "I want to start a new journey today.",
        "input_modality": "text",
    }
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["conversation_id"] is not None
    assert len(data["response"]) > 0

    # Verify conversation now exists in listing
    list_res = client.get("/api/v1/conversations")
    assert any(c["id"] == data["conversation_id"] for c in list_res.json()["conversations"])
