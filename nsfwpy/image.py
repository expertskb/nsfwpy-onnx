"""
Image loading and CPU-optimized preprocessing utilities for NSFWJS.
"""
import io
import urllib.request
from pathlib import Path
from typing import Union, Tuple, List
import numpy as np
from PIL import Image

from .constants import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

ImageInput = Union[str, Path, bytes, Image.Image]


def load_image(image_input: ImageInput) -> Image.Image:
    """
    Load any image input (path, URL, bytes, or PIL Image) into a PIL RGB Image.
    """
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, (str, Path)):
        src = str(image_input)
        if src.startswith(("http://", "https://")):
            req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
            img = Image.open(io.BytesIO(data))
        else:
            img = Image.open(src)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Handle transparency / Alpha channel by alpha matting over white background
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        alpha_img = img.convert("RGBA")
        background = Image.new("RGBA", alpha_img.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(background, alpha_img)
        return composite.convert("RGB")

    return img.convert("RGB")


def preprocess_image(
    image_input: ImageInput,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    input_shape: Union[List[int], Tuple[int, ...], None] = None,
    normalize_type: str = "imagenet",
) -> np.ndarray:
    """
    Preprocess an image for ONNX model inference.
    Supports NCHW [1, 3, H, W] and NHWC [1, H, W, 3] layout formats.
    """
    img = load_image(image_input)
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)

    arr = np.asarray(img, dtype=np.float32)

    # Apply normalization
    if normalize_type == "tf":
        # TensorFlow style: scale to [-1, 1]
        arr = (arr / 127.5) - 1.0
    elif normalize_type == "scale":
        # Scale to [0, 1]
        arr = arr / 255.0
    else:
        # Standard ImageNet mean/std normalization
        arr = arr / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        arr = (arr - mean) / std

    # Check input layout from expected model input shape
    # Default is NCHW: [1, 3, H, W]
    is_nchw = True
    if input_shape is not None and len(input_shape) == 4:
        if input_shape[1] != 3 and input_shape[3] == 3:
            is_nchw = False

    if is_nchw:
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))

    # Add batch dimension -> [1, C, H, W] or [1, H, W, C]
    return np.expand_dims(arr, axis=0)


def preprocess_batch(
    images: List[ImageInput],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    input_shape: Union[List[int], Tuple[int, ...], None] = None,
    normalize_type: str = "imagenet",
) -> np.ndarray:
    """
    Preprocess a list of images into a single batch NumPy tensor.
    """
    preprocessed = [
        preprocess_image(img, target_size, input_shape, normalize_type) for img in images
    ]
    return np.vstack(preprocessed)
