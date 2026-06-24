"""
Backend Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# Model paths
CNN_MODEL_PATH = os.getenv("CNN_MODEL_PATH", "models/trained/cnn_model.pth")
SVM_MODEL_PATH = os.getenv("SVM_MODEL_PATH", "models/trained/svm_model.pkl")

# Training
INPUT_SIZE = (64, 64)
FEATURE_DIM = 128
ENSEMBLE_CNN_WEIGHT = 0.6
ENSEMBLE_SVM_WEIGHT = 0.4
NUM_CLASSES = 5

# Metrics CSV paths
CLIENT1_METRICS = os.getenv("CLIENT1_METRICS", "client1_metrics.csv")
CLIENT2_METRICS = os.getenv("CLIENT2_METRICS", "client2_metrics.csv")

# Demo mode flag
DEMO_MODE = os.getenv("DEMO_MODE", "auto").lower()