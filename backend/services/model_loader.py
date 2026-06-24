"""
Model Loader Service
Loads trained CNN and SVM models with automatic fallback to demo mode
"""
import os
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Global model instances
cnn_model = None
svm_model = None
demo_mode = False


def load_models():
    """
    Load trained CNN and SVM models.
    Falls back to demo mode if models are unavailable.
    Returns: (cnn_model, svm_model, is_demo_mode)
    """
    global cnn_model, svm_model, demo_mode

    # Add project root to path for importing existing models
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Try loading CNN model
    cnn_loaded = _load_cnn(project_root)
    svm_loaded = _load_svm(project_root)

    if cnn_loaded and svm_loaded:
        demo_mode = False
        logger.info("Trained models loaded successfully")
    else:
        demo_mode = True
        logger.info("Models not found - switching to DEMO MODE")

    return cnn_model, svm_model, demo_mode


def _load_cnn(project_root):
    """Load CNN model from saved .pth file"""
    global cnn_model
    try:
        import torch
        from cnn_model import CNNModel

        # Check multiple possible paths
        possible_paths = [
            os.path.join(project_root, "models", "trained", "cnn_model.pth"),
            os.path.join(project_root, "cnn_model.pth"),
            os.path.join(project_root, "checkpoints", "cnn_best.pth"),
        ]

        model_path = None
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break

        if model_path is None:
            logger.warning("CNN model file not found")
            return False

        # Initialize CNN model and load weights
        cnn_model = CNNModel(num_classes=5)
        cnn_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        cnn_model.eval()
        logger.info(f"CNN model loaded from {model_path}")
        return True

    except Exception as e:
        logger.warning(f"Failed to load CNN model: {e}")
        cnn_model = None
        return False


def _load_svm(project_root):
    """Load SVM model from saved .pkl or .joblib file"""
    global svm_model
    try:
        import joblib
        from svm_model import FederatedSVM

        # Check multiple possible paths
        possible_paths = [
            os.path.join(project_root, "models", "trained", "svm_model.pkl"),
            os.path.join(project_root, "models", "trained", "svm_model.joblib"),
            os.path.join(project_root, "svm_model.pkl"),
            os.path.join(project_root, "svm_model.joblib"),
        ]

        model_path = None
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                break

        if model_path is None:
            logger.warning("SVM model file not found")
            return False

        # Try loading with joblib first
        try:
            svm_model = joblib.load(model_path)
            logger.info(f"SVM model loaded from {model_path}")
            return True
        except Exception:
            pass

        # Fall back to pickle
        import pickle
        with open(model_path, "rb") as f:
            svm_model = pickle.load(f)
        logger.info(f"SVM model loaded from {model_path}")
        return True

    except Exception as e:
        logger.warning(f"Failed to load SVM model: {e}")
        svm_model = None
        return False