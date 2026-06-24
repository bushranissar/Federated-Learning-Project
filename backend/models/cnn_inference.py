"""
CNN Model Inference Wrapper
Reuses existing CNNModel architecture from cnn_model.py
"""
import os
import sys
import logging

logger = logging.getLogger(__name__)

# Dynamically add project root
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def get_cnn_model(num_classes=5):
    """
    Get CNN model instance from existing cnn_model.py
    Returns: CNNModel instance
    """
    from cnn_model import CNNModel
    return CNNModel(num_classes=num_classes)


def extract_features(model, image_tensor):
    """
    Extract 128-D feature vector from image using CNN
    Args:
        model: CNNModel instance
        image_tensor: preprocessed image tensor (1, 3, 64, 64)
    Returns:
        features: numpy array of shape (1, 128)
        cnn_probs: numpy array of class probabilities (1, 5)
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        # Extract 128-D features
        features = model.extract_features(image_tensor)

        # Get CNN classification probabilities
        logits = model.forward(image_tensor)
        cnn_probs = F.softmax(logits, dim=1)

    return features.numpy(), cnn_probs.numpy()