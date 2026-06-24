"""
Prediction Router
POST /predict - Upload image and get ensemble prediction
"""
import io
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.services.image_processor import validate_image, preprocess_image
import backend.services.model_loader as model_loader
from backend.services.mock_predictor import MockPredictor
from backend.models.ensemble import ensemble_predict
from backend.utils.constants import DISEASE_INFO

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload tomato leaf image and get disease prediction
    Returns ensemble prediction with CNN, SVM, and combined probabilities
    """
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )

    # Read image bytes
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded."
        )

    # Validate image
    if not validate_image(image_bytes):
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file."
        )

    # Check if in demo mode
    if model_loader.demo_mode or model_loader.cnn_model is None or model_loader.svm_model is None:
        logger.info(
            "Using demo mode for prediction: "
            f"demo_mode={model_loader.demo_mode}, cnn_model_is_none={model_loader.cnn_model is None}, "
            f"svm_model_is_none={model_loader.svm_model is None}"
        )
        result = MockPredictor.get_full_prediction()
    else:
        try:
            import torch
            import torch.nn.functional as F

            # Preprocess image
            image_tensor = preprocess_image(image_bytes)

            # Run inference with no_grad
            with torch.no_grad():
                # Extract CNN features
                cnn_features = model_loader.cnn_model.extract_features(image_tensor)

                # CNN classification probabilities
                cnn_logits = model_loader.cnn_model.forward(image_tensor)
                cnn_probs = F.softmax(cnn_logits, dim=1).cpu().numpy()

                # SVM prediction from CNN features
                features = cnn_features.cpu().numpy()
                svm_pred = model_loader.svm_model.predict(features)
                svm_probs = model_loader.svm_model.predict_proba(features)

            # Ensemble prediction
            result = ensemble_predict(cnn_probs, svm_probs)
            result["success"] = True
            result["demo_mode"] = False

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Fall back to demo mode
            result = MockPredictor.get_full_prediction()
            result["error"] = str(e)

    # Add disease info
    disease_name = result["prediction"]
    if disease_name in DISEASE_INFO:
        result["disease_info"] = DISEASE_INFO[disease_name]
    else:
        result["disease_info"] = None

    return result