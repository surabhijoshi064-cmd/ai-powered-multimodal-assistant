from app.models.conversation import Conversation
from app.models.message import Message
from app.models.document import Document
from app.models.document_chunk import DocumentChunk


def test_create_and_cascade_conversation(db_session):
    # 1. Create Conversation
    conv = Conversation(title="Test Chat")
    db_session.add(conv)
    db_session.commit()
    db_session.refresh(conv)

    assert conv.id is not None
    assert conv.title == "Test Chat"

    # 2. Add Messages
    msg1 = Message(
        conversation_id=conv.id,
        role="user",
        content="Hello, how can I practice box breathing?",
        input_modality="text",
    )
    msg2 = Message(
        conversation_id=conv.id,
        role="assistant",
        content="Box breathing involves inhaling for 4 seconds, holding for 4...",
        input_modality="text",
    )
    db_session.add_all([msg1, msg2])
    db_session.commit()

    # 3. Add Document and Chunks
    doc = Document(
        conversation_id=conv.id,
        original_filename="guide.pdf",
        safe_filename="safe_guide.pdf",
        mime_type="application/pdf",
        file_size=1024,
        processing_status="completed",
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)

    chunk = DocumentChunk(
        document_id=doc.id,
        chunk_index=0,
        chunk_text="Box breathing is a stress relief technique.",
        page_number=1,
    )
    db_session.add(chunk)
    db_session.commit()

    # Verify counts
    assert len(conv.messages) == 2
    assert len(conv.documents) == 1
    assert len(doc.chunks) == 1

    # 4. Delete Conversation and verify cascade
    db_session.delete(conv)
    db_session.commit()

    assert db_session.query(Conversation).filter_by(id=conv.id).first() is None
    assert db_session.query(Message).filter_by(conversation_id=conv.id).count() == 0
    assert db_session.query(Document).filter_by(id=doc.id).first() is None
    assert db_session.query(DocumentChunk).filter_by(document_id=doc.id).count() == 0
