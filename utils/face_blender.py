"""
FaceBlender — seamlessly blends a face patch onto a rendered body image
using Laplacian pyramid blending.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FaceBlender:
    """
    Blends a source face onto a target rendered image using Laplacian pyramids
    for seamless transition at the boundary.

    Args:
        pyramid_levels: Number of Laplacian pyramid levels.
    """

    def __init__(self, pyramid_levels: int = 4):
        self.pyramid_levels = pyramid_levels
        logger.info(f"FaceBlender initialised (pyramid_levels={pyramid_levels})")

    # ------------------------------------------------------------------

    def blend_face_with_identity(
        self,
        rendered_image: Image.Image,
        face_patch: Image.Image,
        face_bbox: Tuple[int, int, int, int],
        landmarks: Optional[List] = None,
        identity_strength: float = 1.0,
    ) -> Image.Image:
        """
        Blend face_patch into rendered_image at the position given by face_bbox.

        Args:
            rendered_image:    Target image (full body render).
            face_patch:        Source face image (already extracted/aligned).
            face_bbox:         (x0, y0, x1, y1) target region on rendered_image.
            landmarks:         (unused) reserved for future alignment.
            identity_strength: Blend weight (0 = only render, 1 = only face).

        Returns:
            Blended PIL.Image with face inserted.
        """
        if face_patch is None or identity_strength == 0.0:
            return rendered_image

        x0, y0, x1, y1 = face_bbox
        target_w = max(x1 - x0, 1)
        target_h = max(y1 - y0, 1)

        # Resize face patch to target region size
        face_resized = face_patch.convert("RGB").resize(
            (target_w, target_h), Image.LANCZOS
        )

        result = rendered_image.copy().convert("RGB")
        rw, rh = result.size

        # Clamp bbox to image bounds
        x0 = max(0, min(x0, rw))
        y0 = max(0, min(y0, rh))
        x1 = max(0, min(x1, rw))
        y1 = max(0, min(y1, rh))

        if x1 <= x0 or y1 <= y0:
            return result

        # Crop existing region from render
        render_region = result.crop((x0, y0, x1, y1))
        rr_w, rr_h = render_region.size
        face_resized = face_resized.resize((rr_w, rr_h), Image.LANCZOS)

        # Laplacian pyramid blend
        blended_region = self._laplacian_blend(
            np.array(face_resized, dtype=np.float32),
            np.array(render_region, dtype=np.float32),
            levels=self.pyramid_levels,
            alpha=identity_strength,
        )
        blended_pil = Image.fromarray(
            np.clip(blended_region, 0, 255).astype(np.uint8)
        )

        result.paste(blended_pil, (x0, y0))
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_gaussian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
        pyramid = [img]
        for _ in range(levels - 1):
            # Simple 2× downsample via average pooling
            h, w = pyramid[-1].shape[:2]
            down = pyramid[-1].reshape(h // 2, 2, w // 2, 2, -1).mean(axis=(1, 3))
            pyramid.append(down)
        return pyramid

    @staticmethod
    def _build_laplacian_pyramid(
        gaussian: List[np.ndarray],
    ) -> List[np.ndarray]:
        laplacian = []
        for i in range(len(gaussian) - 1):
            h, w = gaussian[i].shape[:2]
            up = np.repeat(np.repeat(gaussian[i + 1], 2, axis=0), 2, axis=1)
            up = up[:h, :w]
            laplacian.append(gaussian[i] - up)
        laplacian.append(gaussian[-1])
        return laplacian

    def _laplacian_blend(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        levels: int,
        alpha: float,
    ) -> np.ndarray:
        """Blend src into dst using Laplacian pyramid (constant alpha mask)."""
        # Ensure even dimensions for pyramid building
        h, w = src.shape[:2]
        # Pad to nearest multiple of 2^levels
        pad = 2 ** levels
        h_pad = ((h + pad - 1) // pad) * pad
        w_pad = ((w + pad - 1) // pad) * pad

        def pad_img(img):
            ph = h_pad - img.shape[0]
            pw = w_pad - img.shape[1]
            return np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")

        src_p, dst_p = pad_img(src), pad_img(dst)

        try:
            gp_src = self._build_gaussian_pyramid(src_p, levels)
            gp_dst = self._build_gaussian_pyramid(dst_p, levels)
            lp_src = self._build_laplacian_pyramid(gp_src)
            lp_dst = self._build_laplacian_pyramid(gp_dst)

            blended_lp = [alpha * ls + (1 - alpha) * ld for ls, ld in zip(lp_src, lp_dst)]

            # Reconstruct
            result = blended_lp[-1]
            for lap in reversed(blended_lp[:-1]):
                rh, rw = lap.shape[:2]
                up = np.repeat(np.repeat(result, 2, axis=0), 2, axis=1)
                up = up[:rh, :rw]
                result = up + lap

            return result[:h, :w]

        except Exception as e:
            logger.warning(f"Laplacian blend failed ({e}), falling back to alpha blend")
            return alpha * src + (1 - alpha) * dst
