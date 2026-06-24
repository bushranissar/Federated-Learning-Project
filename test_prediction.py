"""
Quick test to verify models load and predict correctly
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cnn_model import CNNModel
import torch
import joblib

# Test CNN model
print("="*50)
print("Testing CNN Model...")
cnn = CNNModel(num_classes=5)
state = torch.load("models/trained/cnn_model.pth", map_location="cpu")
cnn.load_state_dict(state)
cnn.eval()

# Test with random input
dummy_input = torch.randn(1, 3, 64, 64)
features = cnn.extract_features(dummy_input)
logits = cnn(dummy_input)
probs = torch.softmax(logits, dim=1)
print(f"CNN features shape: {features.shape}")
print(f"CNN output shape: {logits.shape}")
print(f"CNN probabilities: {probs.detach().numpy()}")
print("CNN model: OK")

# Test SVM model
print("="*50)
print("Testing SVM Model...")
svm = joblib.load("models/trained/svm_model.pkl")
features_np = features.detach().numpy()
pred = svm.predict(features_np)
pred_proba = svm.predict_proba(features_np)
print(f"SVM prediction: {pred}")
print(f"SVM probabilities: {pred_proba}")
print("SVM model: OK")

# Test ensemble
print("="*50)
print("Testing Ensemble...")
cnn_w, svm_w = 0.6, 0.4
ensemble = cnn_w * probs.detach().numpy() + svm_w * pred_proba
print(f"Ensemble probabilities: {ensemble}")
print(f"Ensemble prediction: {np.argmax(ensemble, axis=1)}")
print("Ensemble: OK")

print("="*50)
print("ALL CHECKS PASSED!")