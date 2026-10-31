import time
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.config import settings
from app.utils.logging_config import setup_logging
from app.api.routes import health

logger = setup_logging(settings.LOG_LEVEL)

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
    logger.info(f"Started {settings.APP_NAME} in {settings.APP_ENV} mode.")
    yield
    logger.info(f"Stopping {settings.APP_NAME}.")


app = FastAPI(
    title=settings.APP_NAME,
    description="Secure Multimodal AI Chatbot with RAG, Emotional Support, and strict guardrails.",
    version="0.1.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# CORS middleware configured for allowed origins (e.g. Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs incoming requests and execution time safely without sensitive payloads."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} completed with status {response.status_code} in {duration:.3f}s"
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Structured response for validation errors without exposing internal internals."""
    logger.warning(f"Validation error on {request.method} {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation failed",
            "details": [
                {"field": ".".join(str(loc) for loc in err["loc"]), "message": err["msg"]}
                for err in exc.errors()
            ],
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Prevents raw stack traces and internal errors from leaking to users."""
    logger.error(f"Unhandled server error on {request.method} {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "An internal server error occurred. Please try again later or contact support."
        },
    )


from app.api.routes import health, conversations, chat, multimodal

# Include routes
app.include_router(health.router)
app.include_router(conversations.router)
app.include_router(chat.router)
app.include_router(multimodal.router)

