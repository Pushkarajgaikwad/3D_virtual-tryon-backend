"""
Garment texture projection onto a body mesh using orthographic UV mapping.

Strategy
--------
• Classify each vertex as belonging to one of three regions:
    - shirt  : torso + short sleeves (Y between waist and neck, excluding hands)
    - skin   : head, legs, feet, forearms/hands

• Build a 3-section horizontal texture atlas:
    [front_garment | back_garment | skin_patch]

• Map shirt vertices to the garment sections (front/back by Z sign).
• Map skin vertices to the solid skin-tone patch.

This ensures the garment only appears on the body region it should cover,
with the rest of the body rendered in a natural skin tone.
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Neutral skin tone (RGB)
_SKIN_COLOR = np.array([220, 185, 155], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Body-region classification
# ---------------------------------------------------------------------------

def _classify_vertices(
    x_norm: np.ndarray,
    y_norm: np.ndarray,
) -> np.ndarray:
    """
    Return a boolean mask (N,) — True = vertex belongs to the shirt region.

    Uses normalised coordinates where:
        x_norm = 0 → leftmost body edge (left hand tip)
        x_norm = 1 → rightmost body edge (right hand tip)
        y_norm = 0 → bottom of feet
        y_norm = 1 → top of head

    SMPL T-pose approximate landmarks (normalised Y):
        0.00  feet
        0.28  knees
        0.48  waist / hip
        0.65  chest
        0.73  shoulders
        0.80  neck base
        0.85+ head

    Shirt region: waist → neck (y_norm 0.48–0.82), excluding outermost
    10 % of body width (forearms / hands at arm level).
    """
    Y_SHIRT_BOTTOM = 0.48   # waist
    Y_SHIRT_TOP    = 0.82   # just below chin / neck
    X_HAND_MARGIN  = 0.10   # outermost fraction → forearm / hand → skin

    in_shirt_y  = (y_norm >= Y_SHIRT_BOTTOM) & (y_norm <= Y_SHIRT_TOP)
    is_hand     = (x_norm < X_HAND_MARGIN) | (x_norm > (1.0 - X_HAND_MARGIN))

    return in_shirt_y & ~is_hand


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def project_garment_onto_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    garment_texture: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute UV coordinates that map *garment_texture* onto the shirt region
    of *verts*, with the remaining body rendered in a neutral skin tone.

    Parameters
    ----------
    verts : (N, 3) float32
        Body mesh vertices in world space (Y-up, T-pose).
    faces : (F, 3) int
        Face index array (passed through, unused in UV computation).
    garment_texture : (H, W, 3|4) uint8
        Garment image.  Alpha channel is ignored.

    Returns
    -------
    uv_coords : (N, 2) float32
        UV coordinates in [0, 1] for every vertex.
    atlas : (H, 3*W, 3) uint8
        Texture atlas layout:
          cols [0,   W)  → front garment
          cols [W,  2W)  → back garment (horizontally mirrored)
          cols [2W, 3W)  → solid skin-tone patch
    """
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"verts must be (N, 3), got {verts.shape}")

    # ------------------------------------------------------------------ #
    # 1. Build texture atlas  [front | back_mirror | skin]
    # ------------------------------------------------------------------ #
    garment_rgb    = garment_texture[:, :, :3].astype(np.uint8)
    garment_mirror = garment_rgb[:, ::-1, :].copy()

    H, W = garment_rgb.shape[:2]
    skin_patch = np.broadcast_to(_SKIN_COLOR, (H, W, 3)).copy()

    atlas = np.concatenate([garment_rgb, garment_mirror, skin_patch], axis=1)
    # atlas shape: (H, 3*W, 3)

    # ------------------------------------------------------------------ #
    # 2. Normalise vertex positions to [0, 1]
    # ------------------------------------------------------------------ #
    x = verts[:, 0].astype(np.float64)
    y = verts[:, 1].astype(np.float64)
    z = verts[:, 2].astype(np.float64)

    x_range = (x.max() - x.min()) or 1.0
    y_range = (y.max() - y.min()) or 1.0

    x_norm = (x - x.min()) / x_range   # 0 = left,   1 = right
    y_norm = (y - y.min()) / y_range   # 0 = bottom,  1 = top

    # ------------------------------------------------------------------ #
    # 3. Classify vertices: shirt vs skin
    # ------------------------------------------------------------------ #
    is_shirt = _classify_vertices(x_norm, y_norm)
    is_skin  = ~is_shirt

    logger.debug(
        f"Vertex classification: shirt={is_shirt.sum()}, skin={is_skin.sum()}"
    )

    # ------------------------------------------------------------------ #
    # 4. Front / back split (by Z)
    # ------------------------------------------------------------------ #
    z_median = float(np.median(z))
    is_front = z >= z_median

    # ------------------------------------------------------------------ #
    # 5. Compute U coordinate
    #
    #    Atlas sections (each 1/3 of total width):
    #      Section 0  U ∈ [0,   1/3)  front garment
    #      Section 1  U ∈ [1/3, 2/3)  back garment (mirror)
    #      Section 2  U ∈ [2/3, 1  ]  skin patch
    # ------------------------------------------------------------------ #

    # --- Shirt vertices ---
    # Remap x within the shirt's own X extent to fill each garment section.
    # Front: x_norm → [0, 1/3]
    # Back : mirrored x → [1/3, 2/3]
    u_front = x_norm / 3.0
    u_back  = (1.0 / 3.0) + (1.0 - x_norm) / 3.0

    u_shirt = np.where(is_front, u_front, u_back)

    # --- Skin vertices → section 2 ---
    u_skin = (2.0 / 3.0) + x_norm / 3.0

    u = np.where(is_shirt, u_shirt, u_skin)

    # ------------------------------------------------------------------ #
    # 6. Compute V coordinate
    #
    #    For shirt vertices: stretch the garment image to fill the shirt
    #    region (waist→neck maps to V 0→1).
    #    For skin vertices: simple 1−y_norm so texture matches body height.
    # ------------------------------------------------------------------ #
    Y_SHIRT_BOTTOM = 0.48
    Y_SHIRT_TOP    = 0.82
    shirt_y_range  = Y_SHIRT_TOP - Y_SHIRT_BOTTOM

    # V=0 at top (neck), V=1 at bottom (waist)
    v_shirt = (Y_SHIRT_TOP - y_norm) / shirt_y_range
    v_skin  = 1.0 - y_norm

    v = np.where(is_shirt, v_shirt, v_skin)

    # ------------------------------------------------------------------ #
    # 7. Clip and return
    # ------------------------------------------------------------------ #
    u = np.clip(u, 0.0, 1.0).astype(np.float32)
    v = np.clip(v, 0.0, 1.0).astype(np.float32)

    uv_coords = np.stack([u, v], axis=1)   # (N, 2)

    logger.info(
        f"UV projection done: {len(verts)} verts | "
        f"shirt={is_shirt.sum()} back={is_skin.sum()} | "
        f"atlas={atlas.shape}"
    )

    return uv_coords, atlas
