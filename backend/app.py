"""
FastAPI Backend Application
Federated Learning Tomato Leaf Disease Detection Dashboard
"""
import sys
import os
import logging
from contextlib import asynccontextmanager

# Add project root to path for importing existing models
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import HOST, PORT, CORS_ORIGINS
from backend.services.model_loader import load_models
from backend.routers import predict, metrics, federated_status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models on startup"""
    logger.info("Starting backend server...")
    cnn, svm, demo = load_models()
    if demo:
        logger.info("Running in DEMO MODE - no trained models found")
    else:
        logger.info("Running with trained models")
    yield
    logger.info("Shutting down backend server...")


app = FastAPI(
    title="Federated Learning Tomato Leaf Disease Detection API",
    description="Backend API for CNN-SVM ensemble prediction dashboard",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(predict.router, tags=["Prediction"])
app.include_router(metrics.router, tags=["Metrics"])
app.include_router(federated_status.router, tags=["Federated Learning"])


@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "name": "Federated Learning Tomato Leaf Disease Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "POST /predict - Upload image for disease prediction",
            "GET /metrics - Get model performance metrics",
            "GET /federated-status - Get FL training status",
            "GET / - API health check"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host=HOST,
        port=PORT,
        reload=False
    )
