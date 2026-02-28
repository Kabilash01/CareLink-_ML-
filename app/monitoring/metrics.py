"""
CARELINK AI Microservice - Metrics Collector

Production-grade metrics collection for monitoring and observability.
Provides Prometheus-compatible metrics export.
"""

import time
from typing import Dict
from datetime import datetime
from collections import defaultdict
from threading import Lock

from app.core.logging import logger


class MetricsCollector:
    """
    Centralized metrics collection for monitoring.
    
    Collects:
    - Total predictions
    - Predictions by risk level
    - Confidence statistics
    - Rules triggered count
    - Response times
    - Error rates
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.start_time = time.time()
        self.lock = Lock()
        
        # Metrics storage
        self.metrics = {
            "total_predictions": 0,
            "predictions_by_risk": defaultdict(int),
            "total_confidence": 0.0,
            "rules_triggered_count": 0,
            "escalations_count": 0,
            "emergency_flags_count": 0,
            "response_times": [],
            "error_count": 0
        }
        
        logger.info("MetricsCollector initialized")
    
    def record_prediction(
        self,
        prediction: str,
        confidence: float,
        rules_triggered: list,
        escalated: bool = False,
        emergency_flag: bool = False,
        response_time_ms: float = None
    ):
        """
        Record a prediction for metrics.
        
        Args:
            prediction: Risk level prediction
            confidence: Prediction confidence
            rules_triggered: List of triggered rules
            escalated: Whether prediction was escalated
            emergency_flag: Whether emergency flag was set
            response_time_ms: Response time in milliseconds
        """
        with self.lock:
            self.metrics["total_predictions"] += 1
            self.metrics["predictions_by_risk"][prediction] += 1
            self.metrics["total_confidence"] += confidence
            
            if rules_triggered:
                self.metrics["rules_triggered_count"] += len(rules_triggered)
            
            if escalated:
                self.metrics["escalations_count"] += 1
            
            if emergency_flag:
                self.metrics["emergency_flags_count"] += 1
            
            if response_time_ms is not None:
                self.metrics["response_times"].append(response_time_ms)
                # Keep only last 1000 response times
                if len(self.metrics["response_times"]) > 1000:
                    self.metrics["response_times"] = self.metrics["response_times"][-1000:]
    
    def record_error(self):
        """Record an error occurrence."""
        with self.lock:
            self.metrics["error_count"] += 1
    
    def get_metrics(self) -> dict:
        """
        Get current metrics snapshot.
        
        Returns:
            dict: Current metrics
        """
        with self.lock:
            total_predictions = self.metrics["total_predictions"]
            
            # Calculate average confidence
            avg_confidence = (
                self.metrics["total_confidence"] / total_predictions
                if total_predictions > 0 else 0.0
            )
            
            # Calculate average response time
            response_times = self.metrics["response_times"]
            avg_response_time = (
                sum(response_times) / len(response_times)
                if response_times else 0.0
            )
            
            # Calculate uptime
            uptime_seconds = time.time() - self.start_time
            
            return {
                "total_predictions": total_predictions,
                "predictions_by_risk": dict(self.metrics["predictions_by_risk"]),
                "average_confidence": round(avg_confidence, 3),
                "rules_triggered_count": self.metrics["rules_triggered_count"],
                "escalations_count": self.metrics["escalations_count"],
                "emergency_flags_count": self.metrics["emergency_flags_count"],
                "average_response_time_ms": round(avg_response_time, 2),
                "error_count": self.metrics["error_count"],
                "uptime_seconds": round(uptime_seconds, 2),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            str: Prometheus-formatted metrics
        """
        metrics = self.get_metrics()
        
        lines = [
            "# HELP carelink_ai_predictions_total Total number of predictions",
            "# TYPE carelink_ai_predictions_total counter",
            f"carelink_ai_predictions_total {metrics['total_predictions']}",
            "",
            "# HELP carelink_ai_predictions_by_risk Predictions by risk level",
            "# TYPE carelink_ai_predictions_by_risk counter"
        ]
        
        for risk, count in metrics['predictions_by_risk'].items():
            lines.append(f'carelink_ai_predictions_by_risk{{risk="{risk}"}} {count}')
        
        lines.extend([
            "",
            "# HELP carelink_ai_average_confidence Average prediction confidence",
            "# TYPE carelink_ai_average_confidence gauge",
            f"carelink_ai_average_confidence {metrics['average_confidence']}",
            "",
            "# HELP carelink_ai_rules_triggered_total Total rules triggered",
            "# TYPE carelink_ai_rules_triggered_total counter",
            f"carelink_ai_rules_triggered_total {metrics['rules_triggered_count']}",
            "",
            "# HELP carelink_ai_escalations_total Total escalations",
            "# TYPE carelink_ai_escalations_total counter",
            f"carelink_ai_escalations_total {metrics['escalations_count']}",
            "",
            "# HELP carelink_ai_emergency_flags_total Total emergency flags",
            "# TYPE carelink_ai_emergency_flags_total counter",
            f"carelink_ai_emergency_flags_total {metrics['emergency_flags_count']}",
            "",
            "# HELP carelink_ai_response_time_ms Average response time in milliseconds",
            "# TYPE carelink_ai_response_time_ms gauge",
            f"carelink_ai_response_time_ms {metrics['average_response_time_ms']}",
            "",
            "# HELP carelink_ai_errors_total Total errors",
            "# TYPE carelink_ai_errors_total counter",
            f"carelink_ai_errors_total {metrics['error_count']}",
            "",
            "# HELP carelink_ai_uptime_seconds Service uptime in seconds",
            "# TYPE carelink_ai_uptime_seconds counter",
            f"carelink_ai_uptime_seconds {metrics['uptime_seconds']}"
        ])
        
        return "\n".join(lines)
    
    def reset_metrics(self):
        """Reset all metrics (for testing purposes only)."""
        with self.lock:
            self.start_time = time.time()
            self.metrics = {
                "total_predictions": 0,
                "predictions_by_risk": defaultdict(int),
                "total_confidence": 0.0,
                "rules_triggered_count": 0,
                "escalations_count": 0,
                "emergency_flags_count": 0,
                "response_times": [],
                "error_count": 0
            }
            logger.info("Metrics reset")
