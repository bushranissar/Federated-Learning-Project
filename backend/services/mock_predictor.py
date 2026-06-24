"""
Mock Predictor Service
Generates realistic mock predictions when trained models are unavailable
"""
import random
import numpy as np
from backend.utils.constants import DISEASE_CLASSES


# Seed for reproducibility
random.seed(42)
np.random.seed(42)


class MockPredictor:
    """
    Generates realistic mock predictions mimicking:
    - CNN probability distributions
    - SVM probability distributions
    - Ensemble predictions
    - Confidence scores (70-95%)
    """

    @staticmethod
    def generate_probabilities():
        """
        Generate realistic class probabilities.
        One class dominates (the "predicted" class) with 65-95% confidence.
        Remaining classes share the rest.
        """
        n_classes = len(DISEASE_CLASSES)

        # Random dominant class (skewed toward classes 0-2 for realism)
        dominant_idx = random.choices(
            [0, 1, 2, 3, 4],
            weights=[0.3, 0.2, 0.2, 0.15, 0.15],
            k=1
        )[0]

        # Generate probabilities
        probs = np.random.dirichlet(
            np.ones(n_classes) * 0.3
        )

        # Boost dominant class probability (65-95%)
        boost = random.uniform(0.65, 0.95)
        dominant_val = probs[dominant_idx]
        scaling = (1 - boost) / (1 - dominant_val)

        for i in range(n_classes):
            if i == dominant_idx:
                probs[i] = boost
            else:
                probs[i] *= scaling

        # Normalize to ensure sum = 1.0
        probs = probs / probs.sum()

        return probs, dominant_idx

    @staticmethod
    def generate_cnn_probs():
        """Generate mock CNN probabilities"""
        probs, idx = MockPredictor.generate_probabilities()
        confidence = float(probs[idx])
        return probs.tolist(), idx, confidence

    @staticmethod
    def generate_svm_probs():
        """Generate mock SVM probabilities (slightly different from CNN)"""
        probs, idx = MockPredictor.generate_probabilities()
        # SVM is slightly less confident
        probs[idx] *= random.uniform(0.85, 0.98)
        probs = probs / probs.sum()
        confidence = float(probs[idx])
        return probs.tolist(), idx, confidence

    @staticmethod
    def generate_ensemble_prediction(cnn_probs, svm_probs):
        """
        Compute ensemble prediction: 0.6 * CNN + 0.4 * SVM
        """
        cnn_arr = np.array(cnn_probs)
        svm_arr = np.array(svm_probs)

        ensemble_probs = 0.6 * cnn_arr + 0.4 * svm_arr
        predicted_class = int(np.argmax(ensemble_probs))
        confidence = float(ensemble_probs[predicted_class])

        return ensemble_probs.tolist(), predicted_class, confidence

    @staticmethod
    def get_full_prediction():
        """
        Generate complete mock prediction with:
        - CNN probabilities
        - SVM probabilities
        - Ensemble probabilities
        - All predictions
        - All confidences
        """
        # Generate CNN prediction
        cnn_probs, cnn_pred, cnn_conf = MockPredictor.generate_cnn_probs()
        cnn_class = DISEASE_CLASSES[cnn_pred]

        # Generate SVM prediction
        svm_probs, svm_pred, svm_conf = MockPredictor.generate_svm_probs()
        svm_class = DISEASE_CLASSES[svm_pred]

        # Ensemble prediction
        ensemble_probs, ensemble_pred, ensemble_conf = \
            MockPredictor.generate_ensemble_prediction(cnn_probs, svm_probs)
        ensemble_class = DISEASE_CLASSES[ensemble_pred]

        return {
            "success": True,
            "demo_mode": True,
            "prediction": ensemble_class,
            "confidence": ensemble_conf,
            "cnn_prediction": cnn_class,
            "cnn_confidence": cnn_conf,
            "cnn_probabilities": {
                DISEASE_CLASSES[i]: float(cnn_probs[i])
                for i in range(len(DISEASE_CLASSES))
            },
            "svm_prediction": svm_class,
            "svm_confidence": svm_conf,
            "svm_probabilities": {
                DISEASE_CLASSES[i]: float(svm_probs[i])
                for i in range(len(DISEASE_CLASSES))
            },
            "ensemble_prediction": ensemble_class,
            "ensemble_confidence": ensemble_conf,
            "ensemble_probabilities": {
                DISEASE_CLASSES[i]: float(ensemble_probs[i])
                for i in range(len(DISEASE_CLASSES))
            },
            "probabilities": {
                DISEASE_CLASSES[i]: float(ensemble_probs[i])
                for i in range(len(DISEASE_CLASSES))
            },
            "top5": [
                {
                    "class": DISEASE_CLASSES[i],
                    "probability": float(ensemble_probs[i])
                }
                for i in np.argsort(ensemble_probs)[::-1]
            ]
        }


def generate_mock_metrics():
    """Generate realistic mock metrics"""
    return {
        "accuracy": round(random.uniform(0.88, 0.96), 4),
        "precision": round(random.uniform(0.87, 0.95), 4),
        "recall": round(random.uniform(0.86, 0.94), 4),
        "f1_score": round(random.uniform(0.87, 0.95), 4),
        "loss": round(random.uniform(0.08, 0.25), 4),
        "demo_mode": True,
        "classification_report": {
            DISEASE_CLASSES[i]: {
                "precision": round(random.uniform(0.85, 0.98), 4),
                "recall": round(random.uniform(0.85, 0.98), 4),
                "f1_score": round(random.uniform(0.85, 0.98), 4),
                "support": random.randint(50, 200)
            }
            for i in range(len(DISEASE_CLASSES))
        },
        "confusion_matrix": [
            [85, 3, 2, 1, 1],
            [2, 78, 5, 2, 3],
            [1, 4, 82, 3, 2],
            [2, 3, 2, 79, 4],
            [1, 2, 3, 2, 80]
        ]
    }