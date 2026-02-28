"""
CARELINK AI Microservice - Main Application Entry Point

Production-grade FastAPI application for healthcare symptom triage,
risk classification, and clinical safety enforcement.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.routes import router, triage_service
from app.core.logging import setup_logging, logger
from app.core.settings import get_settings
from app.monitoring.metrics import MetricsCollector
from app.db.base import init_db

# Initialize settings
settings = get_settings()

# Use the same MetricsCollector as routes (shared instance via module import)
metrics_collector = MetricsCollector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    Handles model loading and resource cleanup.
    """
    # Startup
    setup_logging()
    logger.info(f"Starting CARELINK AI Microservice v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    try:
        # Initialise SQLite — creates tables if they don't exist
        init_db()
        logger.info("SQLite database initialised")
    except Exception as e:
        logger.error(f"Database init failed: {e}")

    try:
        # Load ML model into the shared triage_service instance used by routes
        await triage_service.model_service.load_model()
        logger.info("ML model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")
        logger.warning("Service will continue with degraded functionality")
    
    yield
    
    # Shutdown
    logger.info("Shutting down CARELINK AI Microservice")


# Create FastAPI application
app = FastAPI(
    title="CARELINK AI Microservice",
    description="Healthcare Symptom Triage and Risk Assessment API",
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Serve frontend static files
_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(_frontend_dir)), name="frontend")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        """Serve the single-page triage UI."""
        return FileResponse(str(_frontend_dir / "index.html"))


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for container orchestration.
    Returns service status and model availability.
    """
    ms = triage_service.model_service
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "carelink-ai-service",
            "version": settings.APP_VERSION,
            "model_loaded": ms.is_loaded,
            "model_version": ms.model_version if ms.is_loaded else None
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
