"""
Image preprocessing utilities for person and garment images.
"""

import logging
from io import BytesIO

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

# Standard ImageNet normalisation used by most pretrained backbones
_PERSON_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_GARMENT_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_person_image(image_bytes: bytes) -> torch.Tensor:
    """
    Decode and normalise a person image.

    Args:
        image_bytes: Raw image bytes (JPEG / PNG).

    Returns:
        Float tensor of shape (1, 3, 224, 224), ImageNet-normalised.
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = _PERSON_TRANSFORM(image)          # (3, 224, 224)
    return tensor.unsqueeze(0)                  # (1, 3, 224, 224)


def preprocess_garment_image(image_bytes: bytes):
    """
    Decode, normalise, and extract a binary mask for a garment image.

    The mask separates the garment from a (typically white/uniform) background
    using a simple luminance threshold.

    Args:
        image_bytes: Raw image bytes (JPEG / PNG).

    Returns:
        tuple:
            tensor  – Float tensor (1, 3, 224, 224), ImageNet-normalised.
            mask    – Float tensor (1, 1, 224, 224), values in {0, 1}.
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # --- Build binary mask from luminance ---
    gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
    # Pixels darker than 0.95 are considered garment (not background)
    mask_np = (gray < 0.95).astype(np.float32)
    mask_resized = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
        (224, 224), Image.NEAREST
    )
    mask_tensor = torch.from_numpy(
        np.array(mask_resized, dtype=np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0)              # (1, 1, 224, 224)

    tensor = _GARMENT_TRANSFORM(image)       # (3, 224, 224)
    return tensor.unsqueeze(0), mask_tensor  # (1, 3, 224, 224), (1, 1, 224, 224)
