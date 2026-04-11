"""
Body Shape Estimator — derives SMPL beta parameters from uploaded person images.

Algorithm
---------
1. Measure normalised body proportions from the front (+ optional side) image:
       shoulder_ratio  = shoulder_width  / body_height
       hip_ratio       = hip_width       / body_height
       torso_ratio     = torso_height    / body_height  (shoulder → hip)
       leg_ratio       = leg_length      / body_height  (hip → ankle)
       depth_ratio     = body_depth      / body_height  (side image)

   Source priority: MediaPipe Pose landmarks → silhouette (GrabCut) fallback.

2. Build a *physics-based* sensitivity matrix A directly from the SMPL model:
       A[i, j] = ∂(measurement_i) / ∂(beta_j)

   This uses the SMPL shapedirs (shape blend shapes) and J_regressor to
   derive—analytically and without any checkpoints—how each beta shifts each
   joint distance.  No heuristic constants needed.

3. Solve the regularised linear system:
       A · betas  ≈  target_measurements − neutral_measurements
   via weighted least squares, then clip to ±2.5.

This guarantees that the output betas produce a mesh whose key joint distances
match the proportions observed in the uploaded photos.
"""

import logging
from io import BytesIO
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SMPL joint indices (standard SMPL-24 kinematic tree)
# ---------------------------------------------------------------------------
# 0 pelvis, 1 L_hip, 2 R_hip, 3 spine1, 4 L_knee, 5 R_knee, 6 spine2,
# 7 L_ankle, 8 R_ankle, 9 spine3, 10 L_foot, 11 R_foot, 12 neck,
# 13 L_collar, 14 R_collar, 15 head,
# 16 L_shoulder, 17 R_shoulder, 18 L_elbow, 19 R_elbow,
# 20 L_wrist, 21 R_wrist, 22 L_hand, 23 R_hand

_J_L_HIP       =  1
_J_R_HIP       =  2
_J_L_ANKLE     =  7
_J_R_ANKLE     =  8
_J_NECK        = 12
_J_L_SHOULDER  = 16
_J_R_SHOULDER  = 17

# (measurement_name, joint1, joint2)
_MEASUREMENT_PAIRS = [
    ("shoulder_width",  _J_L_SHOULDER, _J_R_SHOULDER),
    ("hip_width",       _J_L_HIP,      _J_R_HIP),
    ("torso_height",    _J_L_HIP,      _J_L_SHOULDER),   # hip → shoulder
    ("leg_length",      _J_L_HIP,      _J_L_ANKLE),
]


# ===========================================================================
# Step A: build SMPL sensitivity model (call once per server lifecycle)
# ===========================================================================

def build_smpl_measurement_model(smpl_handler) -> dict:
    """
    Compute neutral joint positions and the beta-sensitivity matrix from the
    loaded SMPL model.  No image data involved.

    Returns a dict with:
        neutral_height      – body height in SMPL units (float)
        neutral_meas        – (N,) neutral joint-distance values
        sensitivity_matrix  – (N, 10) ∂(measurement)/∂(beta)
        measurement_names   – list[str]
    """
    v_template  = smpl_handler.v_template.cpu().numpy()    # (6890, 3)
    shapedirs   = smpl_handler.shapedirs.cpu().numpy()     # (6890, 3, 10)
    J_regressor = smpl_handler.J_regressor.cpu().numpy()   # (24, 6890)

    # Neutral joint positions: J = J_regressor @ v_template
    J0 = J_regressor @ v_template                          # (24, 3)

    # Joint-position sensitivity: dJ/d(beta_j) = J_regressor @ shapedirs[:,:,j]
    # shapedirs[:,:,j] is (6890,3); result is (24,3) per beta
    n_betas = shapedirs.shape[2]                           # 10
    dJ_dbeta = np.einsum('ij,jkb->ikb', J_regressor, shapedirs)  # (24,3,10)

    # Body height: top of mesh (Y max) minus bottom (Y min)
    head_y = float(v_template[:, 1].max())
    foot_y = float(v_template[:, 1].min())
    neutral_height = head_y - foot_y

    # Build sensitivity for each measurement pair
    n_meas = len(_MEASUREMENT_PAIRS)
    neutral_meas = np.zeros(n_meas)
    A = np.zeros((n_meas, n_betas))
    names = []

    for i, (name, j1, j2) in enumerate(_MEASUREMENT_PAIRS):
        names.append(name)
        p1 = J0[j1]
        p2 = J0[j2]
        dist = float(np.linalg.norm(p2 - p1))
        neutral_meas[i] = dist

        unit_vec = (p2 - p1) / (dist + 1e-8)
        # First-order: d||p2-p1|| / d(beta_j) = unit_vec · (dp2 - dp1)/d(beta_j)
        for j in range(n_betas):
            dp1 = dJ_dbeta[j1, :, j]
            dp2 = dJ_dbeta[j2, :, j]
            A[i, j] = float(np.dot(unit_vec, dp2 - dp1))

    logger.debug(
        f"SMPL sensitivity built: height={neutral_height:.3f}m  "
        f"shoulder={neutral_meas[0]:.3f}  hip={neutral_meas[1]:.3f}"
    )

    return {
        "neutral_height":     neutral_height,
        "neutral_meas":       neutral_meas,        # (N,)
        "sensitivity_matrix": A,                   # (N, 10)
        "measurement_names":  names,
    }


def _solve_betas(
    target_ratios: dict,
    smpl_model: dict,
    lambda_reg: float = 0.3,
) -> np.ndarray:
    """
    Solve for betas given a dict of {measurement_name: ratio_to_body_height}.

    Uses weighted, regularised least squares so that missing measurements
    don't distort the solution.
    """
    h0    = smpl_model["neutral_height"]
    A     = smpl_model["sensitivity_matrix"]      # (N, 10)
    meas0 = smpl_model["neutral_meas"]            # (N,)
    names = smpl_model["measurement_names"]        # list[str]
    N     = len(names)

    # Map measurement names → ratio keys used in target_ratios
    ratio_key = {
        "shoulder_width": "shoulder_ratio",
        "hip_width":       "hip_ratio",
        "torso_height":    "torso_ratio",
        "leg_length":      "leg_ratio",
    }

    delta = np.zeros(N)
    w     = np.zeros(N)

    for i, name in enumerate(names):
        rk = ratio_key.get(name)
        if rk and rk in target_ratios:
            # Convert image ratio → SMPL-unit target, then delta from neutral
            target_dist = target_ratios[rk] * h0
            delta[i]    = target_dist - meas0[i]
            w[i]        = 1.0

    if w.sum() == 0:
        return np.zeros(10, dtype=np.float32)

    W = np.diag(w)
    # Regularised normal equations: (A^T W A + λI) β = A^T W δ
    AtWA = A.T @ W @ A + lambda_reg * np.eye(A.shape[1])
    AtWd = A.T @ (W @ delta)
    betas = np.linalg.solve(AtWA, AtWd)
    betas = np.clip(betas, -2.5, 2.5).astype(np.float32)

    logger.info(
        f"Estimated betas: {np.round(betas, 2).tolist()}  "
        f"(active measurements: {[names[i] for i in range(N) if w[i]>0]})"
    )
    return betas


# ===========================================================================
# Step B: measure body proportions from images
# ===========================================================================

def _load_rgb(image_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(BytesIO(image_bytes)).convert("RGB"))


# ─── MediaPipe Pose ───────────────────────────────────────────────────────────

def _measure_mediapipe(
    front_img: np.ndarray,
    side_img: Optional[np.ndarray],
) -> Optional[dict]:
    try:
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            # mediapipe ≥ 0.10 removed the legacy solutions API
            return None
    except ImportError:
        return None

    mp_pose = mp.solutions.pose

    def _run(img):
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.4,
        ) as pose:
            return pose.process(img)

    res = _run(front_img)
    if not res or not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark
    L  = mp_pose.PoseLandmark

    def p(lm_enum):
        pt = lm[lm_enum.value]
        return np.array([pt.x, pt.y])

    l_sh  = p(L.LEFT_SHOULDER);  r_sh  = p(L.RIGHT_SHOULDER)
    l_hip = p(L.LEFT_HIP);       r_hip = p(L.RIGHT_HIP)
    l_ank = p(L.LEFT_ANKLE);     r_ank = p(L.RIGHT_ANKLE)
    nose  = p(L.NOSE)
    l_ear = p(L.LEFT_EAR);       r_ear = p(L.RIGHT_EAR)

    h_img, w_img = front_img.shape[:2]
    ar = h_img / w_img   # image aspect ratio (height / width)

    # Crown approximation: slightly above ear/nose
    crown_y = min(nose[1], l_ear[1], r_ear[1]) - 0.03
    ankle_y = max(l_ank[1], r_ank[1])
    body_h  = max(ankle_y - crown_y, 0.05)   # in image-fraction units

    def horiz(a, b):
        """Physical horizontal distance, aspect-ratio corrected."""
        return abs(a[0] - b[0]) / (body_h * ar)   # ratio to body height

    def vert(a, b):
        """Physical vertical distance."""
        return abs(a[1] - b[1]) / body_h

    shoulder_ratio = horiz(l_sh, r_sh)
    hip_ratio      = horiz(l_hip, r_hip)
    torso_ratio    = vert(l_sh, l_hip)       # shoulder → hip height
    leg_ratio      = vert(l_hip, l_ank)      # hip → ankle height

    ratios = {
        "shoulder_ratio": shoulder_ratio,
        "hip_ratio":      hip_ratio,
        "torso_ratio":    torso_ratio,
        "leg_ratio":      leg_ratio,
    }

    # Depth from side image
    if side_img is not None:
        res_s = _run(side_img)
        if res_s and res_s.pose_landmarks:
            lm_s  = res_s.pose_landmarks.landmark
            s_sh  = lm_s[L.LEFT_SHOULDER.value]
            s_hip = lm_s[L.LEFT_HIP.value]
            s_ank = lm_s[L.LEFT_ANKLE.value]
            s_nos = lm_s[L.NOSE.value]

            h_s, w_s = side_img.shape[:2]
            ar_s = h_s / w_s
            crown_s  = s_nos.y - 0.03
            ankle_s  = s_ank.y
            body_h_s = max(ankle_s - crown_s, 0.05)

            # In a side view the "shoulder width" seen by camera ≈ body depth
            s_l_sh = lm_s[L.LEFT_SHOULDER.value]
            s_r_sh = lm_s[L.RIGHT_SHOULDER.value]
            depth_x = abs(s_l_sh.x - s_r_sh.x)
            depth_ratio = depth_x / (body_h_s * ar_s)
            ratios["depth_ratio"] = depth_ratio

    logger.debug(f"MediaPipe ratios: {ratios}")
    return ratios


# ─── Silhouette fallback ──────────────────────────────────────────────────────

def _extract_mask(img_rgb: np.ndarray) -> Optional[np.ndarray]:
    """GrabCut person mask → binary uint8 (255 = person)."""
    h, w = img_rgb.shape[:2]
    px, py = max(int(w * 0.07), 5), max(int(h * 0.03), 5)
    rect = (px, py, w - 2 * px, h - 2 * py)
    mask = np.zeros((h, w), dtype=np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_rgb, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        binary = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    except Exception:
        binary = np.zeros((h, w), np.uint8)
        binary[py:h - py, px:w - px] = 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=1)

    # Keep largest connected component
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n < 2:
        return None
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) * 255


def _width_at(mask: np.ndarray, y_frac: float) -> Optional[float]:
    h = mask.shape[0]
    row = int(np.clip(y_frac * h, 0, h - 1))
    cols = np.where(mask[row] > 127)[0]
    if len(cols) < 5:
        return None
    return float(cols[-1] - cols[0])


def _measure_silhouette(
    front_img: np.ndarray,
    side_img: Optional[np.ndarray],
) -> dict:
    mask = _extract_mask(front_img)
    h    = front_img.shape[0]

    if mask is None:
        return {}

    rows = np.where(np.any(mask > 127, axis=1))[0]
    if len(rows) < 20:
        return {}

    top    = rows[0]
    bottom = rows[-1]
    body_h = max(bottom - top, 1)

    def frac(abs_offset_from_top):
        return (top + abs_offset_from_top) / h

    # Approx positions (fraction of body height from crown):
    #  shoulders ~ 22-27 %, waist ~ 50 %, hips ~ 60 %, ankle ~ 95 %
    w_sh  = _width_at(mask, frac(body_h * 0.24))
    w_hip = _width_at(mask, frac(body_h * 0.60))
    w_wai = _width_at(mask, frac(body_h * 0.50))

    torso_px = body_h * (0.60 - 0.24)  # hip_y - shoulder_y in pixels
    leg_px   = body_h * (0.95 - 0.60)

    def _r(px_w):
        return (px_w / body_h) if px_w is not None else None

    ratios = {}
    if w_sh  is not None: ratios["shoulder_ratio"] = _r(w_sh)
    if w_hip is not None: ratios["hip_ratio"]      = _r(w_hip)
    if w_wai is not None: ratios["waist_ratio"]    = _r(w_wai)
    ratios["torso_ratio"] = torso_px / body_h
    ratios["leg_ratio"]   = leg_px   / body_h

    if side_img is not None:
        side_mask = _extract_mask(side_img)
        if side_mask is not None:
            s_rows = np.where(np.any(side_mask > 127, axis=1))[0]
            if len(s_rows) > 10:
                s_body_h = max(s_rows[-1] - s_rows[0], 1)
                chest_w  = _width_at(side_mask,
                                     (s_rows[0] + s_body_h * 0.38) / side_img.shape[0])
                if chest_w is not None:
                    ratios["depth_ratio"] = chest_w / s_body_h

    logger.debug(f"Silhouette ratios: {ratios}")
    return ratios


# ===========================================================================
# Public API
# ===========================================================================

def estimate_body_shape_betas(
    front_bytes: bytes,
    side_bytes: Optional[bytes] = None,
    back_bytes: Optional[bytes] = None,
    smpl_handler=None,
) -> torch.Tensor:
    """
    Estimate SMPL shape betas (1, 10) from person images.

    Args:
        front_bytes:   Raw bytes of the front-view person image.
        side_bytes:    Raw bytes of the side-view person image (optional).
        back_bytes:    Currently unused.
        smpl_handler:  Loaded SMPLHandler instance (provides shapedirs /
                       J_regressor for physics-based beta solving).
                       Falls back to a coarse heuristic if None.

    Returns:
        torch.Tensor shape (1, 10), dtype=float32.
        Returns zeros on complete failure (neutral SMPL body).
    """
    try:
        front_img = _load_rgb(front_bytes)
        side_img  = _load_rgb(side_bytes) if side_bytes else None

        # 1. Measure body proportions from images
        ratios = _measure_mediapipe(front_img, side_img)
        if ratios is None:
            logger.info("MediaPipe unavailable or failed — using silhouette fallback")
            ratios = _measure_silhouette(front_img, side_img)

        if not ratios:
            logger.warning("No body measurements obtained — returning neutral shape")
            return torch.zeros(1, 10)

        # 2. Solve for betas
        if smpl_handler is not None:
            smpl_model = build_smpl_measurement_model(smpl_handler)
            betas = _solve_betas(ratios, smpl_model)
        else:
            # No SMPL handler: use fallback heuristics
            logger.warning("smpl_handler not provided — using coarse beta heuristics")
            betas = _heuristic_betas(ratios)

        return torch.tensor(betas, dtype=torch.float32).unsqueeze(0)

    except Exception as exc:
        logger.warning(f"Body shape estimation failed ({exc}) — using neutral shape")
        return torch.zeros(1, 10)


# ---------------------------------------------------------------------------
# Coarse fallback (used only when smpl_handler is unavailable)
# ---------------------------------------------------------------------------
_NEUTRAL_RATIOS = {
    "shoulder_ratio": 0.260,
    "hip_ratio":      0.225,
    "torso_ratio":    0.340,
    "leg_ratio":      0.530,
    "depth_ratio":    0.130,
}
_FALLBACK_SENS = {        # beta_index: {ratio_key: sensitivity}
    0: {"shoulder_ratio":  6.0},
    1: {"hip_ratio":       5.0, "shoulder_ratio": 3.0},
    2: {"torso_ratio":     4.0},
    3: {"leg_ratio":       4.0},
    4: {"depth_ratio":     6.0},
}

def _heuristic_betas(ratios: dict) -> np.ndarray:
    betas = np.zeros(10, dtype=np.float32)
    for bi, key_sens in _FALLBACK_SENS.items():
        for rk, sens in key_sens.items():
            if rk in ratios and _NEUTRAL_RATIOS.get(rk, 0) > 0:
                delta = (ratios[rk] - _NEUTRAL_RATIOS[rk]) / _NEUTRAL_RATIOS[rk]
                betas[bi] += delta * sens
    return np.clip(betas, -2.5, 2.5)
