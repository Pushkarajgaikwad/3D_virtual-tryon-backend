"""
TextureManager — extracts and manages garment textures for UV mapping.
"""

import logging
import os
from io import BytesIO
from typing import Tuple, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class TextureManager:
    """
    Generates UV textures from garment images and loads per-template UV maps.

    Args:
        templates_root: Directory containing template sub-folders.
        uv_res:         Resolution of the generated texture (pixels).
    """

    def __init__(self, templates_root: str = "templates", uv_res: int = 1024):
        self.templates_root = templates_root
        self.uv_res = uv_res
        logger.info(
            f"TextureManager initialised (uv_res={uv_res}, templates_root='{templates_root}')"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_texture(
        self,
        garment_bytes: bytes,
        template_id: str,
        quality: str = "fast",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a flat texture image from the garment photo.

        Steps:
          1. Decode garment image.
          2. Remove background (luminance threshold).
          3. Resize to uv_res × uv_res.

        Args:
            garment_bytes: Raw image bytes.
            template_id:   Template being used (unused here, reserved for future).
            quality:       "fast" or "high".

        Returns:
            texture  – (H, W, 3) uint8 numpy array.
            mask     – (H, W)    float32 numpy array, values in [0, 1].
        """
        res = self.uv_res if quality == "high" else self.uv_res // 2

        img = Image.open(BytesIO(garment_bytes)).convert("RGB")
        img = img.resize((res, res), Image.LANCZOS)
        texture = np.array(img, dtype=np.uint8)

        # Simple luminance-based background removal
        gray = texture.mean(axis=2) / 255.0
        mask = (gray < 0.95).astype(np.float32)

        # Fill background pixels with average garment colour
        if mask.sum() > 0:
            avg_color = texture[mask > 0].mean(axis=0).astype(np.uint8)
        else:
            avg_color = np.array([128, 128, 128], dtype=np.uint8)
        bg_pixels = mask < 0.5
        texture[bg_pixels] = avg_color

        logger.debug(f"Generated texture: {texture.shape}, mask coverage={mask.mean():.2%}")
        return texture, mask

    def load_template_uv(
        self,
        template_id: str,
        mesh: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Load UV coordinates for a template mesh.

        Tries to read  <templates_root>/<template_id>/uv.npy.
        Falls back to spherical projection if file not found.

        Args:
            template_id: Template identifier.
            mesh:        (N, 3) vertex array used for spherical fallback.

        Returns:
            (N, 2) float32 UV coordinate array.
        """
        uv_path = os.path.join(self.templates_root, template_id, "uv.npy")
        if os.path.exists(uv_path):
            uv = np.load(uv_path).astype(np.float32)
            logger.debug(f"Loaded UV from {uv_path}: {uv.shape}")
            return uv

        # Fallback: spherical UV projection
        if mesh is not None:
            uv = _spherical_uv(mesh)
            logger.debug(f"Using spherical UV projection: {uv.shape}")
            return uv

        raise FileNotFoundError(
            f"No UV map found for template '{template_id}' and no mesh provided for fallback."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spherical_uv(vertices: np.ndarray) -> np.ndarray:
    """Generate UV coordinates via spherical projection."""
    vertices = np.asarray(vertices, dtype=np.float32)
    centered = vertices - vertices.mean(axis=0)
    x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
    theta = np.arctan2(x, z + 1e-8)
    phi = np.arctan2(y, np.sqrt(x ** 2 + z ** 2) + 1e-8)
    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi / 2) / np.pi
    uv = np.stack([u, v], axis=1).astype(np.float32)
    return np.clip(uv, 0.0, 1.0)
