"""
SVM Model Inference Wrapper
Reuses existing FederatedSVM from svm_model.py
"""
import os
import sys
import numpy as np

# Dynamically add project root
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def get_svm_model():
    """
    Get SVM model instance from existing svm_model.py
    Returns: FederatedSVM instance
    """
    from svm_model import FederatedSVM
    return FederatedSVM()


def svm_predict(model, features):
    """
    Predict using SVM model
    Args:
        model: FederatedSVM instance
        features: numpy array of shape (1, 128)
    Returns:
        prediction: int (class index)
        probabilities: numpy array of shape (1, 5)
    """
    # Ensure features is 2D
    if features.ndim == 1:
        features = features.reshape(1, -1)

    prediction = model.predict(features)
    probabilities = model.predict_proba(features)

    return int(prediction[0]), probabilities