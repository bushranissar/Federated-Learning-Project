"""
Image Processing Service
Handles loading, validation, and preprocessing of tomato leaf images
"""
import io
from PIL import Image
import torchvision.transforms as transforms
from backend.config import INPUT_SIZE


def validate_image(image_bytes: bytes) -> bool:
    """Validate uploaded image file"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        return True
    except Exception:
        return False


def preprocess_image(image_bytes: bytes):
    """
    Preprocess image for CNN inference
    Returns: normalized tensor of shape (1, 3, H, W)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor