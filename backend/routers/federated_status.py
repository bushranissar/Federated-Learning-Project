"""
Federated Status Router
GET /federated-status - Get federated learning training status
"""
import logging
from fastapi import APIRouter
from backend.services.metrics_aggregator import get_federated_status

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/federated-status")
async def federated_status():
    """
    Get federated learning status including:
    - Number of clients
    - Communication rounds completed
    - FedAvg aggregation status
    - Client-specific performance
    - Accuracy/loss progression over rounds
    """
    result = get_federated_status()
    return result