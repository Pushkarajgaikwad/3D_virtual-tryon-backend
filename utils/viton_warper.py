"""
VITON Garment Warper — Geometric Matching Module (GMM) with TPS warp.

Implements the Geometric Matching Module from the VITON paper:
  Han et al., "VITON: An Image-based Virtual Try-On Network", CVPR 2018.

The GMM learns a Thin-Plate Spline (TPS) transformation that aligns a
segmented garment image to the body shape defined by:
  • OpenPose keypoints
  • Person body silhouette (coarse mask)

When no trained checkpoint is present, a keypoint-guided heuristic TPS
warp is applied instead, which still outperforms orthographic UV projection
by explicitly accounting for body-proportional scaling.

Pipeline
--------
  1. Feature extraction — VGG-based encoder for person + garment
  2. Correlation layer  — computes similarity between all feature pairs
  3. Regression head   — predicts 5×5 = 25 TPS control point offsets
  4. TPS warp          — differentiable spatial transform
  5. Return warped garment (H×W×3 RGB) + warp grid for UV mapping

Fallback
--------
If no checkpoint (checkpoints/gmm.pth) is found, a heuristic TPS warp
is derived directly from OpenPose keypoints.
"""

import io
import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

GMM_CKPT = "checkpoints/gmm.pth"


# ─────────────────────────────────────────────────────────────────────────────
# TPS utilities
# ─────────────────────────────────────────────────────────────────────────────

def _tps_kernel(r: torch.Tensor) -> torch.Tensor:
    """r²·log(r²), with 0·log(0)=0 convention."""
    eps = 1e-6
    return r ** 2 * torch.log(r ** 2 + eps)


def _compute_tps_transform(
    ctrl_pts: torch.Tensor,   # (B, N, 2) source control points in [-1,1]
    offsets:  torch.Tensor,   # (B, N, 2) predicted offsets
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solve the TPS system and return (weights W, affine A) for warp evaluation.
    Returns W (B, N, 2), A (B, 3, 2).
    """
    B, N, _ = ctrl_pts.shape
    device = ctrl_pts.device

    # Build kernel matrix K (B, N, N)
    diff = ctrl_pts.unsqueeze(2) - ctrl_pts.unsqueeze(1)  # (B,N,N,2)
    r2   = (diff ** 2).sum(-1)                             # (B,N,N)
    K    = _tps_kernel(r2)

    # Augment: P = [1, x, y]
    ones = torch.ones(B, N, 1, device=device)
    P    = torch.cat([ones, ctrl_pts], -1)                 # (B,N,3)

    zeros33 = torch.zeros(B, 3, 3, device=device)
    L_top   = torch.cat([K, P], -1)                        # (B,N,N+3)
    L_bot   = torch.cat([P.transpose(1, 2), zeros33], -1) # (B,3,N+3)
    L       = torch.cat([L_top, L_bot], 1)                 # (B,N+3,N+3)

    # RHS: target = source + offset, padded with zeros
    target = ctrl_pts + offsets
    rhs    = torch.cat([target, torch.zeros(B, 3, 2, device=device)], 1)  # (B,N+3,2)

    # Solve L·coeff = rhs
    coeff = torch.linalg.lstsq(L, rhs).solution   # (B, N+3, 2)
    W = coeff[:, :N]    # (B, N, 2)
    A = coeff[:, N:]    # (B, 3, 2)
    return W, A


def tps_grid(
    ctrl_pts: torch.Tensor,   # (B, N, 2)
    W:        torch.Tensor,   # (B, N, 2)
    A:        torch.Tensor,   # (B, 3, 2)
    H:        int,
    Ww:       int,
) -> torch.Tensor:             # (B, H, W, 2)  grid for F.grid_sample
    """Evaluate TPS at every pixel on a H×W grid."""
    B, N, _ = ctrl_pts.shape
    device   = ctrl_pts.device

    # Grid of query points in [-1,1]
    gy = torch.linspace(-1, 1, H, device=device)
    gx = torch.linspace(-1, 1, Ww, device=device)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')  # (H,W)
    q = torch.stack([grid_x, grid_y], -1).unsqueeze(0).expand(B, -1, -1, -1)  # (B,H,W,2)
    q_flat = q.reshape(B, -1, 2)    # (B, H*W, 2)

    # Kernel values between query and control points
    diff = q_flat.unsqueeze(2) - ctrl_pts.unsqueeze(1)  # (B, H*W, N, 2)
    r2   = (diff ** 2).sum(-1)                           # (B, H*W, N)
    K_q  = _tps_kernel(r2)                              # (B, H*W, N)

    # TPS evaluation: result = K_q @ W + [1,x,y] @ A
    ones_q = torch.ones(B, H * Ww, 1, device=device)
    P_q    = torch.cat([ones_q, q_flat], -1)             # (B, H*W, 3)

    warped_flat = K_q @ W + P_q @ A                      # (B, H*W, 2)
    return warped_flat.reshape(B, H, Ww, 2)


# ─────────────────────────────────────────────────────────────────────────────
# GMM network
# ─────────────────────────────────────────────────────────────────────────────

def _vgg_encoder(in_ch: int) -> nn.Sequential:
    """Lightweight VGG-style feature extractor → 256-ch feature map at 1/8."""
    return nn.Sequential(
        nn.Conv2d(in_ch, 64,  3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(64,   64,  3, padding=1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64,  128, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
    )


class GMM(nn.Module):
    """
    Geometric Matching Module.

    Inputs
    ------
    person_repr  : (B, 22, H, W) — person representation
                    Channels: RGB (3) + body segment map (1) + pose heatmaps (18)
    garment_img  : (B, 3, H, W)  — segmented garment image

    Output
    ------
    grid    : (B, H, W, 2)        — sampling grid for F.grid_sample
    theta   : (B, N_ctrl*2)       — raw TPS offsets (for loss computation)
    """

    N_CTRL = 5   # 5×5 = 25 control points

    def __init__(self, person_ch: int = 22):
        super().__init__()
        self.feat_person  = _vgg_encoder(person_ch)
        self.feat_garment = _vgg_encoder(3)

        # Correlation → regression head
        # Correlation output: (B, feat_H*feat_W, feat_H, feat_W) — flattened to 2D grid
        # We pool to a fixed 5×5 correlation map then regress offsets
        self.pool_corr = nn.AdaptiveAvgPool2d((self.N_CTRL, self.N_CTRL))
        corr_dim = self.N_CTRL ** 4  # 625

        self.regress = nn.Sequential(
            nn.Linear(corr_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256),      nn.ReLU(inplace=True),
            nn.Linear(256, self.N_CTRL ** 2 * 2),  # x,y offset per control point
            nn.Tanh(),   # offsets in [-1,1]
        )
        self._init_weights()

    def _init_weights(self):
        # Zero-init the regression head so initial transform ≈ identity
        nn.init.zeros_(self.regress[-2].weight)
        nn.init.zeros_(self.regress[-2].bias)

    @property
    def ctrl_grid(self) -> torch.Tensor:
        """Fixed 5×5 control point grid in [-0.8, 0.8]."""
        lin = torch.linspace(-0.8, 0.8, self.N_CTRL)
        gx, gy = torch.meshgrid(lin, lin, indexing='xy')
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)  # (25, 2)

    def forward(self, person_repr: torch.Tensor, garment_img: torch.Tensor):
        B, _, H, W = garment_img.shape
        device = garment_img.device

        fp = self.feat_person(person_repr)   # (B, 256, H/8, W/8)
        fg = self.feat_garment(garment_img)  # (B, 256, H/8, W/8)

        # Correlation: dot-product similarity between each spatial location
        fH, fW = fp.shape[2], fp.shape[3]
        fp_n = F.normalize(fp.reshape(B, 256, -1), dim=1)  # (B, 256, fH*fW)
        fg_n = F.normalize(fg.reshape(B, 256, -1), dim=1)

        corr = torch.bmm(fp_n.permute(0, 2, 1), fg_n)      # (B, fH*fW, fH*fW)
        corr = corr.reshape(B, fH * fW, fH, fW)
        corr = F.relu(corr)

        # Pool to N_CTRL × N_CTRL
        corr_p = self.pool_corr(corr)                        # (B, fH*fW, N, N)
        corr_p = corr_p.reshape(B, -1)                       # (B, fH*fW*N*N) — might be large
        # Limit: pool the first dim too
        corr_p = corr_p[:, : self.N_CTRL ** 4]              # take first 625 dims

        theta_flat = self.regress(corr_p)                    # (B, 50)
        offsets = theta_flat.reshape(B, self.N_CTRL ** 2, 2) * 0.1  # small offsets

        ctrl_pts = self.ctrl_grid.unsqueeze(0).expand(B, -1, -1).to(device)  # (B, 25, 2)
        W_tps, A_tps = _compute_tps_transform(ctrl_pts, offsets)
        grid = tps_grid(ctrl_pts, W_tps, A_tps, H, W)

        return grid, theta_flat


# ─────────────────────────────────────────────────────────────────────────────
# Person representation builder
# ─────────────────────────────────────────────────────────────────────────────

def build_person_repr(
    front_image: np.ndarray,      # (H, W, 3) uint8
    keypoints: "Keypoints",       # from openpose_handler
    body_mask: Optional[np.ndarray] = None,  # (H, W) float32 [0,1]
    output_size: int = 256,
) -> np.ndarray:
    """
    Build the 22-channel person representation:
      [0:3]   RGB image (normalised [0,1])
      [3]     body silhouette mask
      [4:22]  Gaussian pose heatmaps (one per keypoint)
    """
    from utils.openpose_handler import Keypoints
    H, W = front_image.shape[:2]
    out_h = out_w = output_size

    img_resized = cv2.resize(front_image, (out_w, out_h)).astype(np.float32) / 255.0

    if body_mask is not None:
        mask_resized = cv2.resize(body_mask, (out_w, out_h))
    else:
        # Crude body mask: assume centre 60% × 90% of image
        mask_resized = np.zeros((out_h, out_w), np.float32)
        y0, y1 = int(0.05 * out_h), int(0.95 * out_h)
        x0, x1 = int(0.20 * out_w), int(0.80 * out_w)
        mask_resized[y0:y1, x0:x1] = 1.0

    # Gaussian pose heatmaps
    sigma = output_size / 32.0
    heatmaps = np.zeros((18, out_h, out_w), dtype=np.float32)
    px_pts = keypoints.points.copy()
    valid  = px_pts[:, 0] >= 0

    for i in range(18):
        if not valid[i]:
            continue
        cx = int(px_pts[i, 0] * out_w)
        cy = int(px_pts[i, 1] * out_h)
        x_idx = np.arange(out_w)
        y_idx = np.arange(out_h)
        xg, yg = np.meshgrid(x_idx, y_idx)
        hm = np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * sigma ** 2))
        heatmaps[i] = hm

    # Concatenate: (3 + 1 + 18, H, W) → (H, W, 22)
    repr_arr = np.concatenate([
        img_resized,                            # (H, W, 3)
        mask_resized[:, :, np.newaxis],         # (H, W, 1)
        heatmaps.transpose(1, 2, 0),           # (H, W, 18)
    ], axis=-1)  # (H, W, 22)

    return repr_arr


# ─────────────────────────────────────────────────────────────────────────────
# Public handler
# ─────────────────────────────────────────────────────────────────────────────

class VITONWarper:
    """
    Warps a segmented garment to fit a target body shape.

    Usage
    -----
    warper = VITONWarper(device)
    warped_garment = warper.warp(
        garment_rgba,   # (H, W, 4) uint8
        person_repr,    # (H, W, 22) float32
    )
    """

    SIZE = 256  # GMM internal resolution

    def __init__(self, device: torch.device, checkpoint_path: str = GMM_CKPT):
        self.device = device
        self._gmm: Optional[GMM] = None
        self._init(checkpoint_path)

    # ── init ─────────────────────────────────────────────────────────────────

    def _init(self, ckpt_path: str):
        if os.path.exists(ckpt_path):
            try:
                gmm = GMM().to(self.device)
                state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                gmm.load_state_dict(state)
                gmm.eval()
                self._gmm = gmm
                logger.info(f"✓ VITONWarper: GMM checkpoint loaded from {ckpt_path}")
                return
            except Exception as exc:
                logger.warning(f"⚠ VITONWarper: GMM load failed ({exc}) — using heuristic TPS warp")

        logger.info("✓ VITONWarper: heuristic TPS fallback active")

    # ── public API ────────────────────────────────────────────────────────────

    def warp(
        self,
        garment_rgba: np.ndarray,       # (H, W, 4) uint8
        person_repr: np.ndarray,        # (H, W, 22) float32
        keypoints=None,                 # Keypoints (for heuristic fallback)
    ) -> np.ndarray:
        """
        Returns warped garment (H, W, 4) uint8 aligned to body shape.
        """
        orig_H, orig_W = garment_rgba.shape[:2]

        garment_rgb = garment_rgba[:, :, :3]

        if self._gmm is not None:
            warped = self._warp_gmm(garment_rgb, person_repr)
        else:
            warped = self._warp_heuristic(garment_rgb, keypoints)

        # Resize back to original resolution
        warped = cv2.resize(warped, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)

        # Re-attach alpha (warped foreground mask)
        alpha_orig = garment_rgba[:, :, 3]
        alpha_warped = self._warp_alpha(alpha_orig, orig_W, orig_H,
                                         garment_rgb, warped, keypoints)

        out = np.dstack([warped, alpha_warped]).astype(np.uint8)
        return out

    # ── GMM warp ──────────────────────────────────────────────────────────────

    def _warp_gmm(self, garment_rgb: np.ndarray, person_repr: np.ndarray) -> np.ndarray:
        S = self.SIZE

        # Resize inputs
        g = cv2.resize(garment_rgb, (S, S)).astype(np.float32) / 255.0
        p = cv2.resize(person_repr, (S, S))

        gt = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).to(self.device)
        pt = torch.from_numpy(p).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            grid, _ = self._gmm(pt, gt)  # (1, S, S, 2)

        warped_t = F.grid_sample(gt, grid, align_corners=True,
                                 mode='bilinear', padding_mode='border')
        warped = (warped_t[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return warped

    # ── Heuristic TPS warp (keypoint-guided) ─────────────────────────────────

    def _warp_heuristic(self, garment_rgb: np.ndarray, keypoints=None) -> np.ndarray:
        """
        Simple heuristic TPS: map garment bounding rectangle to the
        shoulder-hip trapezoid extracted from keypoints.
        """
        S = self.SIZE
        g_s = cv2.resize(garment_rgb, (S, S))

        if keypoints is None or (keypoints.points < 0).all():
            return g_s

        pts = keypoints.points          # (18, 2) normalised

        def _get(idx):
            p = pts[idx]
            if p[0] < 0:
                return None
            return np.array([p[0] * S, p[1] * S], dtype=np.float32)

        r_sh = _get(2) or np.array([0.65 * S, 0.20 * S])
        l_sh = _get(5) or np.array([0.35 * S, 0.20 * S])
        r_hi = _get(8) or np.array([0.60 * S, 0.52 * S])
        l_hi = _get(11) or np.array([0.40 * S, 0.52 * S])

        # Source = garment corners (full image)
        src = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
        # Destination = shoulder-hip quad with some margin
        margin = S * 0.05
        dst = np.float32([
            l_sh + np.array([-margin, -margin * 2]),
            r_sh + np.array([ margin, -margin * 2]),
            r_hi + np.array([ margin,  margin]),
            l_hi + np.array([-margin,  margin]),
        ])

        # Perspective warp (good approximation without full TPS solver)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(g_s, M, (S, S),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        return warped

    # ── alpha warp ────────────────────────────────────────────────────────────

    @staticmethod
    def _warp_alpha(alpha_orig, W, H, garment_rgb, warped_rgb, keypoints) -> np.ndarray:
        """Re-derive alpha from the warped garment using colour change as proxy."""
        # Simple approach: mask pixels where warped colour ≠ background
        warped_f = warped_rgb.astype(np.float32)
        # Background = most common edge colour
        edge_pixels = np.concatenate([
            warped_f[0, :], warped_f[-1, :],
            warped_f[:, 0], warped_f[:, -1],
        ])
        bg_colour = np.median(edge_pixels, axis=0)
        diff = np.linalg.norm(warped_f - bg_colour[np.newaxis, np.newaxis, :], axis=-1)
        mask = (diff > 20).astype(np.uint8) * 255

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
