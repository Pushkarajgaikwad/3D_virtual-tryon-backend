"""
TextureWarpEngine — warps a flat garment texture onto a deformed 3D mesh.
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TextureWarpEngine:
    """
    Maps a 2-D garment texture onto a deformed garment mesh using UV coordinates,
    producing per-vertex colours and a warped texture image.

    Args:
        texture_resolution: Output texture resolution (pixels per side).
    """

    def __init__(self, texture_resolution: int = 1024):
        self.texture_resolution = texture_resolution
        logger.info(f"TextureWarpEngine initialised (resolution={texture_resolution})")

    # ------------------------------------------------------------------

    def warp_texture_to_mesh(
        self,
        texture: np.ndarray,
        uv_map: np.ndarray,
        draped_verts: np.ndarray,
        template_faces,
        quality: str = "fast",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample texture at UV coordinates to produce per-vertex colours and a
        warped texture image.

        Args:
            texture:        (H, W, 3) uint8 garment texture.
            uv_map:         (N, 2) float32 UV coordinates, values in [0, 1].
            draped_verts:   (N, 3) float32 deformed vertex positions (unused here
                            but kept for future geometry-aware warping).
            template_faces: (F, 3) int face index array.
            quality:        "fast" or "high".

        Returns:
            vertex_colors  – (N, 3) float32 array, values in [0, 1].
            warped_texture – (H, W, 3) uint8 re-packed texture.
        """
        uv_map = np.clip(np.asarray(uv_map, dtype=np.float32), 0.0, 1.0)
        H, W = texture.shape[:2]
        N = uv_map.shape[0]

        # Sample texture at each UV coordinate (nearest-neighbour)
        pixel_x = np.floor(uv_map[:, 0] * (W - 1)).astype(np.int32)
        pixel_y = np.floor((1.0 - uv_map[:, 1]) * (H - 1)).astype(np.int32)
        pixel_x = np.clip(pixel_x, 0, W - 1)
        pixel_y = np.clip(pixel_y, 0, H - 1)

        sampled = texture[pixel_y, pixel_x]  # (N, 3) uint8
        vertex_colors = sampled.astype(np.float32) / 255.0

        # Build a warped texture by scattering vertex colours back to a grid
        res = self.texture_resolution if quality == "high" else self.texture_resolution // 2
        warped = np.zeros((res, res, 3), dtype=np.uint8)
        tx = np.floor(uv_map[:, 0] * (res - 1)).astype(np.int32)
        ty = np.floor((1.0 - uv_map[:, 1]) * (res - 1)).astype(np.int32)
        tx = np.clip(tx, 0, res - 1)
        ty = np.clip(ty, 0, res - 1)
        warped[ty, tx] = sampled

        logger.debug(
            f"Texture warp complete: {N} verts, vertex_colors {vertex_colors.shape}, "
            f"warped {warped.shape}"
        )
        return vertex_colors, warped
