"""
CARELINK AI Microservice - Audit Logger

Production-grade audit logging for compliance and traceability.
Logs all predictions for HIPAA compliance and quality monitoring.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from app.core.logging import logger
from app.core.settings import get_settings

settings = get_settings()


class AuditLogger:
    """
    Audit logging for prediction traceability.
    
    Logs:
    - All predictions with full context
    - User actions
    - Model versions
    - Rule triggers
    - System events
    
    For production deployment, integrate with:
    - Database (PostgreSQL, MongoDB)
    - Cloud logging (CloudWatch, Stackdriver)
    - SIEM systems
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize audit logger.
        
        Args:
            log_dir: Directory for audit logs (default: logs/)
        """
        self.enabled = settings.ENABLE_AUDIT_LOGGING
        self.log_dir = log_dir or Path("logs")
        
        if self.enabled:
            self._ensure_log_directory()
        
        logger.info(f"AuditLogger initialized (enabled={self.enabled})")
    
    def _ensure_log_directory(self):
        """Ensure log directory exists."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create log directory: {str(e)}")
            self.enabled = False
    
    def log_prediction(
        self,
        request_id: str,
        input_data: Dict[str, Any],
        prediction: str,
        confidence: float,
        probabilities: Dict[str, float],
        rules_triggered: list,
        model_version: str,
        escalated: bool,
        emergency_flag: bool,
        processing_time_ms: float,
        client_ip: Optional[str] = None
    ):
        """
        Log a prediction for audit trail.
        
        Args:
            request_id: Unique request ID
            input_data: Input data (symptoms, age, etc.)
            prediction: Final prediction
            confidence: Prediction confidence
            probabilities: Full probability distribution
            rules_triggered: Triggered rules
            model_version: Model version
            escalated: Escalation flag
            emergency_flag: Emergency flag
            processing_time_ms: Processing time
            client_ip: Client IP address
        """
        if not self.enabled:
            return
        
        audit_entry = {
            "event_type": "prediction",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": request_id,
            "input": {
                "symptoms_text": input_data.get("symptoms_text", ""),
                "age": input_data.get("age"),
                "duration_days": input_data.get("duration_days"),
                "chronic_conditions": input_data.get("chronic_conditions", []),
                "language": input_data.get("language", "en")
            },
            "output": {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": probabilities,
                "escalated": escalated,
                "emergency_flag": emergency_flag
            },
            "processing": {
                "model_version": model_version,
                "rules_triggered": rules_triggered,
                "processing_time_ms": processing_time_ms
            },
            "metadata": {
                "client_ip": client_ip,
                "service_version": settings.APP_VERSION
            }
        }
        
        self._write_audit_log(audit_entry)
    
    def log_model_load(self, model_version: str, success: bool, error: Optional[str] = None):
        """
        Log model loading event.
        
        Args:
            model_version: Model version
            success: Success flag
            error: Error message if failed
        """
        if not self.enabled:
            return
        
        audit_entry = {
            "event_type": "model_load",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_version": model_version,
            "success": success,
            "error": error
        }
        
        self._write_audit_log(audit_entry)
    
    def log_error(
        self,
        request_id: str,
        error_type: str,
        error_message: str,
        context: Dict[str, Any]
    ):
        """
        Log error for audit trail.
        
        Args:
            request_id: Request ID
            error_type: Error type
            error_message: Error message
            context: Additional context
        """
        if not self.enabled:
            return
        
        audit_entry = {
            "event_type": "error",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": request_id,
            "error_type": error_type,
            "error_message": error_message,
            "context": context
        }
        
        self._write_audit_log(audit_entry)
    
    def _write_audit_log(self, entry: Dict[str, Any]):
        """
        Write audit log entry to file.
        
        In production, send to:
        - Database for querying
        - Cloud logging service
        - SIEM system
        
        Args:
            entry: Audit log entry
        """
        try:
            # Use daily log files
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            log_file = self.log_dir / f"audit_{date_str}.jsonl"
            
            # Write as JSON line
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {str(e)}")
    
    def query_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        event_type: Optional[str] = None
    ) -> list:
        """
        Query audit logs (simple file-based implementation).
        
        In production, use database queries.
        
        Args:
            start_date: Start date
            end_date: End date
            event_type: Filter by event type
            
        Returns:
            list: Matching audit entries
        """
        if not self.enabled:
            return []
        
        results = []
        
        # Iterate through log files in date range
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            log_file = self.log_dir / f"audit_{date_str}.jsonl"
            
            if log_file.exists():
                try:
                    with open(log_file, "r") as f:
                        for line in f:
                            entry = json.loads(line)
                            
                            # Filter by event type if specified
                            if event_type and entry.get("event_type") != event_type:
                                continue
                            
                            results.append(entry)
                            
                except Exception as e:
                    logger.error(f"Failed to read audit log {log_file}: {str(e)}")
            
            # Move to next day
            from datetime import timedelta
            current_date += timedelta(days=1)
        
        return results
