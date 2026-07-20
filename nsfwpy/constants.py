"""
Constants and configuration defaults for NSFWPY in Python.
"""
from typing import List, Dict

# Standard NSFWJS 5 categories
NSFW_CATEGORIES: List[str] = [
    "Drawing",
    "Hentai",
    "Neutral",
    "Porn",
    "Sexy",
]

# Image preprocessing defaults
DEFAULT_IMAGE_SIZE = (224, 224)

# Available local pre-built ONNX models
DEFAULT_MODEL_PATHS: Dict[str, str] = {
    "mobilenet_v2": "models/MobileNet-v2.onnx",
    "mobilenet_v3": "models/MobileNet-v3.onnx",
    "inception_v3": "models/Inception-v3.onnx",
}

# Hugging Face model download URLs if local model is missing
MODEL_DOWNLOAD_URLS: Dict[str, str] = {
    "mobilenet_v2": "https://huggingface.co/expertskb/nsfwpy/resolve/main/MobileNet-v2.onnx?download=true",
    "mobilenet_v3": "https://huggingface.co/expertskb/nsfwpy/resolve/main/MobileNet-v3.onnx?download=true",
    "inception_v3": "https://huggingface.co/expertskb/nsfwpy/resolve/main/Inception-v3.onnx?download=true",
}

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
