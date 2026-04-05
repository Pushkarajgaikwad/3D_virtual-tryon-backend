"""
Garment background removal and texture preparation.

Strategy:
  1. White/light background detection  (pixels where R,G,B > 230 → background)
  2. GrabCut refinement around a central rectangle
  3. Morphological clean-up (close small holes, open speckles)
  4. Fill background with average garment colour for texture use
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes → BGR numpy uint8 array (H, W, 3)."""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        # Fallback: try via PIL
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def _white_background_mask(bgr: np.ndarray, threshold: int = 230) -> np.ndarray:
    """
    Return a boolean mask (H, W) where True = background.

    A pixel is considered background when ALL three channels exceed *threshold*.
    """
    return (
        (bgr[:, :, 0] > threshold) &
        (bgr[:, :, 1] > threshold) &
        (bgr[:, :, 2] > threshold)
    )


def _grabcut_refine(bgr: np.ndarray, initial_bg_mask: np.ndarray) -> np.ndarray:
    """
    Use GrabCut seeded from *initial_bg_mask* to refine the foreground region.

    Returns a uint8 mask (H, W) where 0 = background, 255 = foreground.
    """
    h, w = bgr.shape[:2]

    # Build the GrabCut init mask
    #   cv2.GC_BGD   = 0  (definite background)
    #   cv2.GC_FGD   = 1  (definite foreground)
    #   cv2.GC_PR_BGD= 2  (probable background)
    #   cv2.GC_PR_FGD= 3  (probable foreground)
    gc_mask = np.where(initial_bg_mask, cv2.GC_BGD, cv2.GC_PR_FGD).astype(np.uint8)

    # Centre rectangle: top-left 15 %, bottom-right 85 %
    rect = (
        int(w * 0.15),
        int(h * 0.15),
        int(w * 0.70),
        int(h * 0.70),
    )

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            bgr, gc_mask, rect,
            bgd_model, fgd_model,
            iterCount=5,
            mode=cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as exc:
        logger.warning(f"GrabCut failed ({exc}); using initial mask only")
        return (~initial_bg_mask).astype(np.uint8) * 255

    # Pixels marked as FGD or PR_FGD are foreground
    fg_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        np.uint8(255),
        np.uint8(0),
    )
    return fg_mask


def _morphological_cleanup(mask: np.ndarray) -> np.ndarray:
    """Close small holes and remove isolated noise from the foreground mask."""
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=1)
    return mask


def _average_garment_colour(bgr: np.ndarray, fg_mask: np.ndarray) -> tuple:
    """Return the average BGR colour of the foreground pixels."""
    fg_pixels = bgr[fg_mask > 0]
    if len(fg_pixels) == 0:
        return (128, 128, 128)
    mean_bgr = fg_pixels.mean(axis=0).astype(int)
    return tuple(int(c) for c in mean_bgr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def remove_garment_background(image_bytes: bytes) -> np.ndarray:
    """
    Remove background from a garment image.

    Steps:
      a) Detect white / near-white background (R, G, B > 230).
      b) Refine with GrabCut.
      c) Morphological close/open to clean edges.
      d) Fill background with average garment colour (NOT transparent) so
         that the result can be used directly as a texture.

    Returns:
        RGBA numpy uint8 array (H, W, 4).
        The alpha channel is 255 for foreground, 0 for background.
        Background RGB is filled with the average garment colour.
    """
    bgr = _decode_image(image_bytes)
    h, w = bgr.shape[:2]

    # Step (a): white background
    initial_bg = _white_background_mask(bgr, threshold=230)

    # Step (b): GrabCut — only makes sense when we have both FG and BG pixels
    fg_ratio = 1.0 - initial_bg.mean()
    if 0.05 < fg_ratio < 0.95:
        fg_mask = _grabcut_refine(bgr, initial_bg)
    else:
        fg_mask = (~initial_bg).astype(np.uint8) * 255

    # Step (c): clean up
    fg_mask = _morphological_cleanup(fg_mask)

    # Step (d): fill background with average garment colour
    avg_b, avg_g, avg_r = _average_garment_colour(bgr, fg_mask)
    out_bgr = bgr.copy()
    out_bgr[fg_mask == 0] = [avg_b, avg_g, avg_r]

    # Build RGBA (H, W, 4)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    alpha    = fg_mask  # 255 foreground, 0 background
    rgba     = np.dstack([out_rgb, alpha])
    return rgba.astype(np.uint8)


def prepare_garment_texture(image_bytes: bytes, size: int = 1024) -> np.ndarray:
    """
    Remove background, fill with average garment colour, and resize to a
    square *size × size* texture.

    Returns:
        RGB uint8 numpy array (size, size, 3).
    """
    rgba = remove_garment_background(image_bytes)

    # Where alpha == 0 (background), we already have the average colour from
    # remove_garment_background.  Just take RGB.
    rgb  = rgba[:, :, :3]

    # Resize with LANCZOS (high quality)
    pil  = Image.fromarray(rgb, mode='RGB')
    pil  = pil.resize((size, size), Image.LANCZOS)

    return np.array(pil, dtype=np.uint8)
