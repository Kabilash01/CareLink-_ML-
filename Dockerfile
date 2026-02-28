# Production-grade Dockerfile for CARELINK AI Microservice
# Multi-stage build for optimized image size

# Stage 1: Builder
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Stage 2: Runtime
FROM python:3.11-slim

# Set metadata
LABEL maintainer="CARELINK Engineering Team"
LABEL description="CARELINK AI Microservice for Healthcare Symptom Triage"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ENVIRONMENT=production

# Create non-root user
RUN useradd -m -u 1000 carelink && \
    mkdir -p /app /app/logs && \
    chown -R carelink:carelink /app

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder --chown=carelink:carelink /root/.local /home/carelink/.local

# Copy spaCy model from builder
COPY --from=builder --chown=carelink:carelink /usr/local/lib/python3.11/site-packages/en_core_web_sm* /usr/local/lib/python3.11/site-packages/

# Update PATH
ENV PATH=/home/carelink/.local/bin:$PATH

# Copy application code
COPY --chown=carelink:carelink app/ ./app/
COPY --chown=carelink:carelink training/ ./training/
COPY --chown=carelink:carelink data/ ./data/

# Switch to non-root user
USER carelink

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Run application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
