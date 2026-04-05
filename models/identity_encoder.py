"""
Identity Encoder for Virtual Try-On Backend.

Handles:
- Face embedding extraction (512-dimensional)
- ArcFace-compatible identity representation
- Lightweight pretrained model loading
- Face preprocessing and normalization
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Optional
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


class IdentityEncoder(nn.Module):
    """
    Lightweight face identity encoder.

    Extracts 512-dimensional embeddings from face images using a
    ResNet-18 backbone + identity projection head.

    Architecture:
    - Backbone: ResNet-18 (pretrained on ImageNet)
    - Output layer: 512-D embedding space (L2-normalised)
    """

    def __init__(self, embedding_dim: int = 512):
        super(IdentityEncoder, self).__init__()
        self.embedding_dim = embedding_dim

        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        feature_dim = 512  # ResNet-18 penultimate layer
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        logger.info(f"IdentityEncoder initialised: ResNet-18 → {embedding_dim}-D embedding")

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tensor: (B, 3, H, W) float tensor, ImageNet-normalised.

        Returns:
            (B, embedding_dim) unit-norm embedding tensor.
        """
        features = self.backbone(image_tensor)          # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)
        embedding = self.projection(features)            # (B, embedding_dim)
        return torch.nn.functional.normalize(embedding, p=2, dim=1)

    def extract_identity(self, face_image: Image.Image) -> np.ndarray:
        """
        Extract 512-D identity embedding from a PIL face image.

        Returns:
            (512,) float32 numpy array with unit norm.
        """
        preprocess = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        if face_image.mode != "RGB":
            face_image = face_image.convert("RGB")

        tensor = preprocess(face_image).unsqueeze(0).to("cpu")

        with torch.no_grad():
            embedding = self.forward(tensor)

        return embedding.cpu().numpy().squeeze()

    def extract_identity_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.extract_identity(image)


# ---------------------------------------------------------------------------
# Singleton helper
# ---------------------------------------------------------------------------

_instance: Optional[IdentityEncoder] = None


def get_identity_encoder(embedding_dim: int = 512) -> IdentityEncoder:
    global _instance
    if _instance is None:
        _instance = IdentityEncoder(embedding_dim=embedding_dim)
        _instance.eval()
    return _instance
