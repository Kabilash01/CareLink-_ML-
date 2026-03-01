"""
CARELINK AI Microservice - API Routes

RESTful API endpoints for healthcare symptom triage and risk assessment.
All routes are production-ready with proper error handling and validation.
"""

import io

from fastapi import APIRouter, Depends, HTTPException, File, Form, UploadFile, status, Request
from fastapi.responses import PlainTextResponse

from app.api.schemas import (
    TriageRequest,
    TriageResponse,
    ErrorResponse,
    MetricsResponse
)
from app.api.dependencies import (
    get_request_id,
    check_rate_limit,
    validate_request,
    RequestTimer,
    get_timer
)
from app.services.triage_service import TriageService
from app.monitoring.metrics import MetricsCollector
from app.core.logging import logger, log_request, log_response, log_error
from app.core.settings import get_settings

# Initialize
settings = get_settings()
router = APIRouter()
triage_service = TriageService()
metrics_collector = MetricsCollector()


@router.post(
    "/triage/predict",
    response_model=TriageResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict risk level from symptoms",
    description="Analyze patient symptoms and return risk classification with explanation",
    tags=["Triage"],
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def predict_risk(
    request_data: TriageRequest,
    request: Request,
    request_id: str = Depends(get_request_id),
    timer: RequestTimer = Depends(get_timer),
    _: None = Depends(check_rate_limit),
    context: dict = Depends(validate_request)
):
    """
    Predict risk level from patient symptoms.
    
    This endpoint processes natural language symptom descriptions and returns:
    - Risk classification (LOW/MEDIUM/HIGH)
    - Confidence scores
    - Clinical rule triggers
    - Patient-friendly explanation
    - Model version for traceability
    
    Args:
        request_data: Patient symptom data
        request_id: Unique request identifier
        timer: Request timer for performance tracking
        context: Validated request context
        
    Returns:
        TriageResponse: Risk prediction with explanation
        
    Raises:
        HTTPException: If prediction fails
    """
    
    log_request(
        endpoint="/triage/predict",
        method="POST",
        request_id=request_id,
        symptoms_length=len(request_data.symptoms_text),
        age=request_data.age,
        language=request_data.language
    )
    
    try:
        # Process triage request
        response = await triage_service.predict(
            symptoms_text=request_data.symptoms_text,
            age=request_data.age,
            duration_days=request_data.duration_days,
            chronic_conditions=request_data.chronic_conditions,
            language=request_data.language.value,
            request_id=request_id
        )
        
        # Record metrics
        metrics_collector.record_prediction(
            prediction=response.prediction.value,
            confidence=response.confidence,
            rules_triggered=response.rules_triggered
        )
        
        # Log response
        duration_ms = timer.get_duration_ms()
        log_response(
            endpoint="/triage/predict",
            method="POST",
            request_id=request_id,
            status_code=200,
            duration_ms=duration_ms
        )
        
        return response
        
    except ValueError as e:
        # Input validation error
        logger.warning(f"Validation error: {str(e)}", extra={"request_id": request_id})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        # Unexpected error
        log_error(e, context={"request_id": request_id, "endpoint": "/triage/predict"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request. Please try again."
        )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get service metrics",
    description="Retrieve operational metrics for monitoring",
    tags=["Monitoring"]
)
async def get_metrics():
    """
    Get service metrics.
    
    Returns aggregated metrics including:
    - Total predictions
    - Predictions by risk level
    - Average confidence
    - Rules triggered count
    - Service uptime
    
    Returns:
        MetricsResponse: Service metrics
    """
    try:
        metrics = metrics_collector.get_metrics()
        return metrics
    except Exception as e:
        log_error(e, context={"endpoint": "/metrics"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )


@router.get(
    "/metrics/prometheus",
    response_class=PlainTextResponse,
    summary="Get Prometheus metrics",
    description="Retrieve metrics in Prometheus format",
    tags=["Monitoring"]
)
async def get_prometheus_metrics():
    """
    Get metrics in Prometheus format.
    
    Returns metrics formatted for Prometheus scraping.
    
    Returns:
        str: Prometheus-formatted metrics
    """
    if not settings.ENABLE_METRICS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics endpoint is disabled"
        )
    
    try:
        return metrics_collector.export_prometheus()
    except Exception as e:
        log_error(e, context={"endpoint": "/metrics/prometheus"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export metrics"
        )


@router.get(
    "/model/info",
    summary="Get model information",
    description="Retrieve current model version and metadata",
    tags=["Model"]
)
async def get_model_info():
    """
    Get current model information.
    
    Returns:
        dict: Model metadata including version, training date, and performance metrics
    """
    model_service = triage_service.model_service
    
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_version": model_service.model_version,
        "metadata": model_service.metadata,
        "loaded_at": model_service.loaded_at.isoformat() if model_service.loaded_at else None
    }


@router.post(
    "/model/reload",
    summary="Retry loading the MedGemma model",
    description="Trigger a manual retry of MedGemma model loading if it failed at startup.",
    tags=["Model"],
)
async def reload_model():
    """
    Retry loading MedGemma if the initial load failed (e.g. CUDA not ready at boot).
    """
    explanation_svc = triage_service.explanation_service
    msg = explanation_svc.retry_load()
    return {
        "message": msg,
        "model_ready": explanation_svc.model_ready,
    }


@router.post(
    "/triage/analyze-image",
    summary="Analyze a medical image with MedGemma",
    description=(
        "Upload a medical image (chest X-ray, skin photo, wound, fundus, etc.) and get "
        "an AI-assisted analysis from MedGemma's vision-language model. "
        "NOT a diagnosis — always consult a healthcare professional."
    ),
    tags=["Triage"],
    responses={
        200: {"description": "Image analysis result"},
        400: {"description": "Invalid image format"},
        503: {"description": "MedGemma not ready"},
    },
)
async def analyze_medical_image(
    image: UploadFile = File(..., description="Medical image file (JPEG/PNG/WEBP/BMP)"),
    question: str = Form(
        default="What do you observe in this medical image? Are there any concerning findings?",
        description="Clinical question to ask about the image",
    ),
    language: str = Form(default="en", description="Response language code: en / es / fr"),
):
    """
    Analyze a medical image using MedGemma vision-language capabilities.

    Suitable image types:
    - Chest X-rays
    - Dermatology / skin lesions
    - Wound or injury photos (taken with camera)
    - Fundus / ophthalmology images
    - Lab report or document scans

    Returns:
        dict: analysis text, model_ready flag, model_name
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pillow is not installed. Run: pip install Pillow>=10.0.0",
        )

    # Validate content type
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
    if image.content_type and image.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image type: {image.content_type}. Use JPEG, PNG, WEBP, or BMP.",
        )

    try:
        image_bytes = await image.read()
        pil_image   = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode image: {exc}",
        )

    explanation_svc = triage_service.explanation_service
    model_ready     = explanation_svc.model_ready

    analysis = explanation_svc.analyze_image(
        image=pil_image,
        question=question,
        language=language,
    )

    logger.info(
        f"Image analysis complete | filename={image.filename} "
        f"size={len(image_bytes)} bytes model_ready={model_ready}"
    )

    return {
        "analysis":   analysis,
        "model_ready": model_ready,
        "model_name":  settings.LLM_HF_MODEL_ID,
        "filename":   image.filename,
    }


@router.get(
    "/rules",
    summary="Get clinical safety rules",
    description="Retrieve list of active clinical safety rules",
    tags=["Rules"]
)
async def get_rules():
    """
    Get active clinical safety rules.
    
    Returns:
        dict: List of active safety rules and their configurations
    """
    from app.services.rule_engine import RuleEngine
    
    rule_engine = RuleEngine()
    
    return {
        "rules_enabled": settings.SAFETY_RULES_ENABLED,
        "rules": rule_engine.get_all_rules(),
        "version": "1.0.0"
    }


@router.post(
    "/triage/chat",
    summary="Follow-up chat after triage",
    description=(
        "Send a follow-up question about the triage result. "
        "Include conversation history and the original triage context for continuity."
    ),
    tags=["Triage"],
    responses={
        200: {"description": "Chat reply"},
        503: {"description": "MedGemma not ready"},
    },
)
async def triage_chat(request: Request):
    """
    Follow-up chat powered by MedGemma.

    Body JSON:
        message (str): User's latest message.
        history (list): Previous turns [{"role":"user"|"assistant","content":"..."}].
        triage_context (dict, optional): {risk_level, symptoms_text, explanation, ...}.
        language (str): Language code (default "en").

    Returns:
        dict: {reply: str, model_ready: bool}
    """
    body = await request.json()
    message = (body.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    history = body.get("history", [])
    triage_context = body.get("triage_context")
    language = body.get("language", "en")

    explanation_svc = triage_service.explanation_service

    reply = explanation_svc.chat(
        message=message,
        history=history,
        triage_context=triage_context,
        language=language,
    )

    return {
        "reply": reply,
        "model_ready": explanation_svc.model_ready,
    }
