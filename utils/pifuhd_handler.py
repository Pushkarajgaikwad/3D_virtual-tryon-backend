"""
PIFuHD Handler — High-resolution implicit 3D human reconstruction.

Implements Pixel-Aligned Implicit Functions (PIFu / PIFuHD) to reconstruct
a detailed clothed human mesh from a single RGB image.

Pipeline
--------
1. Extract multi-scale image features via HG (stacked hourglass) encoder
2. For each 3-D sample point, project onto image plane → sample features
3. MLP predicts inside/outside probability (occupancy)
4. Marching cubes extracts an iso-surface mesh at threshold 0.5

Fallback
--------
If the PIFuHD checkpoint (checkpoints/pifuhd_final.pt) is absent or
cannot be loaded, reconstruction falls back to the existing SMPLHandler.

Reference
---------
Saito et al., "Multi-Level Pixel-Aligned Implicit Function for
High-Resolution 3D Human Digitization", CVPR 2020.
"""

import io
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

PIFUHD_CKPT = "checkpoints/pifuhd_final.pt"
_VOXEL_RES   = 128    # marching-cubes voxel grid resolution
_SAMPLE_BATCH = 10_000  # points per inference batch (memory safety)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class _HourglassBlock(nn.Module):
    """Single hourglass stage."""

    def __init__(self, in_ch: int, mid_ch: int, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.down = nn.ModuleList()
        self.up   = nn.ModuleList()
        self.skip = nn.ModuleList()

        ch = in_ch
        for _ in range(depth):
            self.down.append(nn.Sequential(
                nn.Conv2d(ch, mid_ch, 3, stride=2, padding=1),
                nn.GroupNorm(8, mid_ch), nn.ReLU(inplace=True),
            ))
            self.skip.append(_ResBlock(mid_ch))
            self.up.append(nn.Sequential(
                nn.ConvTranspose2d(mid_ch, mid_ch if _ > 0 else in_ch,
                                   4, stride=2, padding=1),
                nn.GroupNorm(8, mid_ch if _ > 0 else in_ch),
                nn.ReLU(inplace=True),
            ))
            ch = mid_ch

        self.bottleneck = _ResBlock(mid_ch)

    def forward(self, x):
        skips, feat = [], x
        for d, s in zip(self.down, self.skip):
            feat = d(feat)
            skips.append(s(feat))
        feat = self.bottleneck(feat)
        for u, sk in zip(reversed(self.up), reversed(skips)):
            feat = u(feat + sk)
        return feat   # same spatial size as input


class _ImageEncoder(nn.Module):
    """
    Two-level hourglass image encoder.
    Input : (B, 3, H, W)
    Output: coarse features (B, 64, H/4, W/4)
            fine   features (B, 32, H/2, W/2)
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
        )  # → 1/4 res, 64-ch
        self.hg_coarse = _HourglassBlock(64, 64, depth=3)
        self.proj_coarse = nn.Conv2d(64, 64, 1)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
        )  # → 1/2 res, 32-ch
        self.hg_fine = _HourglassBlock(32, 32, depth=2)
        self.proj_fine = nn.Conv2d(32, 32, 1)

    def forward(self, x: torch.Tensor):
        s = self.stem(x)
        coarse = self.proj_coarse(self.hg_coarse(s))
        fine   = self.proj_fine(self.hg_fine(self.upsample(coarse)))
        return coarse, fine   # (B,64,H/4,W/4), (B,32,H/2,W/2)


class _OccupancyMLP(nn.Module):
    """
    MLP that maps (pixel-aligned features + 3-D depth z) → occupancy ∈ [0,1].
    Input dim = coarse_ch + fine_ch + 1  (64+32+1 = 97)
    """

    def __init__(self, feat_dim: int = 97, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),  nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),       nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PIFuHDNet(nn.Module):
    """Full PIFuHD inference network."""

    COARSE_CH = 64
    FINE_CH   = 32

    def __init__(self):
        super().__init__()
        self.encoder = _ImageEncoder()
        self.mlp = _OccupancyMLP(self.COARSE_CH + self.FINE_CH + 1)

    @torch.no_grad()
    def query_occupancy(
        self,
        coarse_feat: torch.Tensor,  # (1, 64, Hc, Wc)
        fine_feat: torch.Tensor,    # (1, 32, Hf, Wf)
        points_3d: torch.Tensor,    # (N, 3)  in NDC [-1,1]
    ) -> torch.Tensor:              # (N,)  occupancy [0,1]
        """
        Sample pixel-aligned features for each 3-D point and predict occupancy.
        """
        # Project X,Y to image plane (assume orthographic, Z = depth)
        xy = points_3d[:, :2].unsqueeze(0).unsqueeze(0)   # (1,1,N,2)
        zd = points_3d[:, 2:3]                             # (N,1)

        c_feat = F.grid_sample(coarse_feat, xy, align_corners=True,
                               mode='bilinear', padding_mode='border')
        c_feat = c_feat.squeeze(0).squeeze(1).T             # (N, 64)

        f_feat = F.grid_sample(fine_feat, xy, align_corners=True,
                               mode='bilinear', padding_mode='border')
        f_feat = f_feat.squeeze(0).squeeze(1).T             # (N, 32)

        feats = torch.cat([c_feat, f_feat, zd.to(c_feat)], dim=-1)  # (N, 97)
        return self.mlp(feats)   # (N,)

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


# ─────────────────────────────────────────────────────────────────────────────
# Public handler
# ─────────────────────────────────────────────────────────────────────────────

class PIFuHDHandler:
    """
    Reconstructs a 3-D human mesh from a single front-facing RGB image.

    Usage
    -----
    handler = PIFuHDHandler(device, smpl_handler)
    verts, faces = handler.reconstruct(image_bytes)  # numpy arrays
    """

    def __init__(self, device: torch.device, smpl_handler=None,
                 checkpoint_path: str = PIFUHD_CKPT):
        self.device = device
        self.smpl_handler = smpl_handler
        self._net: Optional[PIFuHDNet] = None
        self._init(checkpoint_path)

    # ── init ─────────────────────────────────────────────────────────────────

    def _init(self, ckpt_path: str):
        if os.path.exists(ckpt_path):
            try:
                net = PIFuHDNet().to(self.device)
                state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                net.load_state_dict(state)
                net.eval()
                self._net = net
                logger.info(f"✓ PIFuHDHandler: checkpoint loaded from {ckpt_path}")
                return
            except Exception as exc:
                logger.warning(f"⚠ PIFuHDHandler: checkpoint load failed ({exc}) — falling back to SMPL")

        if self.smpl_handler is not None:
            logger.info("✓ PIFuHDHandler: SMPL fallback active")
        else:
            logger.warning("⚠ PIFuHDHandler: no PIFuHD checkpoint and no SMPL handler — mesh will be a sphere")

    # ── public API ────────────────────────────────────────────────────────────

    def reconstruct(
        self,
        image_bytes: bytes,
        keypoints=None,         # optional Keypoints from OpenPoseHandler
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct a 3-D mesh from image bytes.

        Returns
        -------
        verts : (N, 3) float32 numpy array
        faces : (M, 3) int32   numpy array
        """
        if self._net is not None:
            return self._reconstruct_pifuhd(image_bytes)
        return self._reconstruct_smpl(keypoints)

    # ── PIFuHD inference path ─────────────────────────────────────────────────

    def _reconstruct_pifuhd(self, image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from skimage import measure as skm
        except ImportError:
            logger.warning("scikit-image not installed — cannot run marching cubes; falling back to SMPL")
            return self._reconstruct_smpl(None)

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Resize to standard PIFuHD input (512-wide, keep aspect)
        W, H = img.size
        new_w = 512
        new_h = int(H * new_w / W)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Normalise [-1, 1]
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            coarse_feat, fine_feat = self._net(x)

        # Build voxel grid in NDC [-1,1]^3 at resolution _VOXEL_RES
        R = _VOXEL_RES
        lin = torch.linspace(-1, 1, R, device=self.device)
        grid_z, grid_y, grid_x = torch.meshgrid(lin, lin, lin, indexing='ij')
        pts = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], -1)  # (R^3, 3)

        # Query occupancy in batches
        occ = torch.zeros(pts.shape[0], device=self.device)
        for start in range(0, pts.shape[0], _SAMPLE_BATCH):
            batch = pts[start:start + _SAMPLE_BATCH]
            occ[start:start + _SAMPLE_BATCH] = self._net.query_occupancy(
                coarse_feat, fine_feat, batch
            )

        occ_vol = occ.reshape(R, R, R).cpu().numpy()   # (R, R, R)

        # Marching cubes
        verts, faces, _, _ = skm.marching_cubes(occ_vol, level=0.5)
        # Normalise verts to [-1,1]
        verts = verts / (R - 1) * 2.0 - 1.0
        verts = verts.astype(np.float32)
        faces = faces.astype(np.int32)

        logger.info(f"PIFuHD mesh: {verts.shape[0]} verts, {faces.shape[0]} faces")
        return verts, faces

    # ── SMPL fallback ─────────────────────────────────────────────────────────

    def _reconstruct_smpl(self, keypoints) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        if self.smpl_handler is None:
            return self._unit_sphere_mesh()

        # Use keypoints to derive rough body shape (betas = 0 = neutral)
        shape_params = torch.zeros(1, 10)
        pose_params  = torch.zeros(1, 72)

        body_mesh = self.smpl_handler.get_smpl_mesh(shape_params, pose_params)
        verts = body_mesh.verts_list()[0].cpu().numpy().astype(np.float32)
        faces = body_mesh.faces_list()[0].cpu().numpy().astype(np.int32)
        return verts, faces

    # ── last-resort sphere mesh ───────────────────────────────────────────────

    @staticmethod
    def _unit_sphere_mesh(subdivisions: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Create an icosphere as a placeholder mesh."""
        try:
            import trimesh
            sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
            return sphere.vertices.astype(np.float32), sphere.faces.astype(np.int32)
        except Exception:
            # Minimal hand-crafted sphere
            t = (1.0 + 5.0 ** 0.5) / 2.0
            verts = np.array([
                [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
            ], dtype=np.float32)
            verts /= np.linalg.norm(verts, axis=1, keepdims=True)
            faces = np.array([
                [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
                [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
                [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
                [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
            ], dtype=np.int32)
            return verts, faces
