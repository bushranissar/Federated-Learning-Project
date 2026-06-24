import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.services.model_loader import cnn_model, svm_model, demo_mode
print("demo_mode:", demo_mode)
print("cnn_model:", cnn_model)
print("svm_model:", svm_model)
print("cnn is None:", cnn_model is None)
print("svm is None:", svm_model is None)