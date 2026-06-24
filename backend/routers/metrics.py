"""
Metrics Router
GET /metrics - Get model performance metrics
"""
import logging
from fastapi import APIRouter
from backend.services.metrics_aggregator import get_metrics

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/metrics")
async def metrics():
    """
    Get model performance metrics including:
    - Accuracy, Precision, Recall, F1 Score
    - Confusion Matrix
    - Classification Report
    - Training history (round-by-round data)
    """
    result = get_metrics()
    return result