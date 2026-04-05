"""
GarmentClassifier — predicts which template category best matches a garment image.
"""

import logging
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class GarmentClassifier(nn.Module):
    """
    Lightweight CNN classifier that maps a garment image to a template category.

    Architecture:
        MobileNetV3-Small backbone → global average pool → Linear(num_classes)

    Args:
        num_classes: Number of template categories.
        emb_dim:     Embedding dimension (unused in forward, kept for API compat).
    """

    def __init__(self, num_classes: int = 2, emb_dim: int = 16):
        super(GarmentClassifier, self).__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim

        try:
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier[-1] = nn.Linear(in_features, num_classes)
            self.model = backbone
        except Exception:
            # Minimal fallback: two conv layers + linear
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(32, num_classes),
            )

        logger.info(f"GarmentClassifier initialised: {num_classes} classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) float tensor.

        Returns:
            (B, num_classes) logits.
        """
        return self.model(x)

    def load(self, path: str, device: torch.device) -> None:
        """Load model weights from a checkpoint file."""
        state = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state)
        logger.info(f"GarmentClassifier weights loaded from {path}")

    def predict_category(
        self,
        garment_bytes: bytes,
        template_manager,
        device: torch.device,
    ) -> Tuple[str, float]:
        """
        Predict the best template ID for a garment image.

        Args:
            garment_bytes:    Raw image bytes.
            template_manager: TemplateManager instance (for class→ID mapping).
            device:           Inference device.

        Returns:
            (template_id, confidence) tuple.
        """
        image = Image.open(BytesIO(garment_bytes)).convert("RGB")
        tensor = _TRANSFORM(image).unsqueeze(0).to(device)

        self.eval()
        with torch.no_grad():
            logits = self.forward(tensor)                    # (1, num_classes)
            probs = torch.softmax(logits, dim=1)[0]         # (num_classes,)
            class_idx = int(probs.argmax().item())
            confidence = float(probs[class_idx].item())

        templates = template_manager.list_templates()
        if not templates:
            raise ValueError("No templates available in TemplateManager")

        # Map class index to template ID (wrap if fewer templates than classes)
        template_id = templates[class_idx % len(templates)]
        return template_id, confidence
