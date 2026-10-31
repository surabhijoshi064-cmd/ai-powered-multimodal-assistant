import os
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.database import get_db
from app.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", status_code=status.HTTP_200_OK)
def check_health(db: Session = Depends(get_db)):
    """
    Returns application health and basic dependency status without exposing secrets.
    """
    # 1. Database check
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "unavailable"

    # 2. Storage check
    storage_status = "accessible"
    try:
        test_file = os.path.join(settings.STORAGE_DIR, ".health_check")
        with open(test_file, "w") as f:
            f.write("ok")
        if os.path.exists(test_file):
            os.remove(test_file)
    except Exception:
        storage_status = "inaccessible"

    # 3. Overall status
    is_healthy = db_status == "connected" and storage_status == "accessible"

    return {
        "status": "healthy" if is_healthy else "degraded",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "dependencies": {
            "database": db_status,
            "storage": storage_status,
            "configured_providers": {
                "llm": settings.LLM_PROVIDER,
                "vision": settings.VISION_PROVIDER,
                "transcription": settings.TRANSCRIPTION_PROVIDER,
                "tts": settings.TTS_PROVIDER,
                "embedding": settings.EMBEDDING_PROVIDER,
            },
        },
    }
