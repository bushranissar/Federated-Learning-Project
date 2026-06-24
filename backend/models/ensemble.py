"""
Ensemble Prediction Model
Combines CNN and SVM predictions using weighted average
CNN weight: 0.6, SVM weight: 0.4
"""
import numpy as np
from backend.utils.constants import DISEASE_CLASSES
from backend.config import ENSEMBLE_CNN_WEIGHT, ENSEMBLE_SVM_WEIGHT


def ensemble_predict(cnn_probs, svm_probs):
    """
    Compute weighted ensemble prediction
    Args:
        cnn_probs: numpy array of CNN probabilities (1, 5) or (5,)
        svm_probs: numpy array of SVM probabilities (1, 5) or (5,)
    Returns:
        dict with prediction, confidence, and all probabilities
    """
    # Convert to 1D arrays
    if cnn_probs.ndim > 1:
        cnn_probs = cnn_probs.flatten()
    if svm_probs.ndim > 1:
        svm_probs = svm_probs.flatten()

    # Weighted ensemble
    ensemble_probs = (ENSEMBLE_CNN_WEIGHT * cnn_probs +
                      ENSEMBLE_SVM_WEIGHT * svm_probs)

    # Normalize
    ensemble_probs = ensemble_probs / ensemble_probs.sum()

    # Get prediction
    predicted_idx = int(np.argmax(ensemble_probs))
    confidence = float(ensemble_probs[predicted_idx])
    predicted_class = DISEASE_CLASSES[predicted_idx]

    # Get CNN and SVM individual predictions
    cnn_idx = int(np.argmax(cnn_probs))
    svm_idx = int(np.argmax(svm_probs))

    # Build top-5
    top5_indices = np.argsort(ensemble_probs)[::-1][:5]
    top5 = [
        {
            "class": DISEASE_CLASSES[i],
            "probability": float(ensemble_probs[i])
        }
        for i in top5_indices
    ]

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "prediction_index": predicted_idx,
        "cnn_prediction": DISEASE_CLASSES[cnn_idx],
        "cnn_confidence": float(cnn_probs[cnn_idx]),
        "cnn_probabilities": {
            DISEASE_CLASSES[i]: float(cnn_probs[i])
            for i in range(len(DISEASE_CLASSES))
        },
        "svm_prediction": DISEASE_CLASSES[svm_idx],
        "svm_confidence": float(svm_probs[svm_idx]),
        "svm_probabilities": {
            DISEASE_CLASSES[i]: float(svm_probs[i])
            for i in range(len(DISEASE_CLASSES))
        },
        "ensemble_prediction": predicted_class,
        "ensemble_confidence": confidence,
        "ensemble_probabilities": {
            DISEASE_CLASSES[i]: float(ensemble_probs[i])
            for i in range(len(DISEASE_CLASSES))
        },
        "probabilities": {
            DISEASE_CLASSES[i]: float(ensemble_probs[i])
            for i in range(len(DISEASE_CLASSES))
        },
        "top5": top5
    }