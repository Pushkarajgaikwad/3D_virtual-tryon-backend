"""
SMPL body model handler.

Loads SMPL_NEUTRAL.pkl directly (no smplx / chumpy required) and
runs a pure-PyTorch Linear Blend Skinning forward pass.

SMPL forward pass:
  1. Shape blend shapes  →  v_shaped  (6890, 3)
  2. Joint regression    →  J         (24, 3)
  3. Pose blend shapes   →  v_posed   (6890, 3)
  4. LBS skinning        →  vertices  (6890, 3)
"""

import logging
import os
import pickle
import sys
import types
import warnings

import numpy as np
import torch
from utils.mesh_types import Meshes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Inject a minimal chumpy stub so pickle can deserialise old .pkl
# ---------------------------------------------------------------------------

def _inject_chumpy_stub():
    """Register fake chumpy modules in sys.modules before any pickle.load."""
    if 'chumpy' in sys.modules:
        return

    class _Ch:
        """Minimal chumpy.Ch that stores its value as a numpy array."""
        __slots__ = ('_v',)

        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)

        def __init__(self, x=None, *args, **kwargs):
            if x is None:
                self._v = np.array([], dtype=np.float64)
            elif isinstance(x, _Ch):
                self._v = x._v
            elif isinstance(x, np.ndarray):
                self._v = x
            else:
                try:
                    self._v = np.array(x, dtype=np.float64)
                except Exception:
                    self._v = np.array([], dtype=np.float64)

        # Called by pickle for __setstate__
        def __setstate__(self, state):
            if isinstance(state, dict):
                if 'x' in state:
                    v = state['x']
                    self._v = np.array(v) if not isinstance(v, np.ndarray) else v
                elif '_v' in state:
                    self._v = state['_v']
                else:
                    self._v = np.array([], dtype=np.float64)
            elif isinstance(state, np.ndarray):
                self._v = state
            else:
                self._v = np.array([], dtype=np.float64)

        @property
        def r(self):
            return self._v

        def __array__(self, dtype=None, copy=False):
            return self._v.astype(dtype) if dtype else self._v

        @property
        def shape(self):
            return self._v.shape

        @property
        def T(self):
            return self._v.T

        def __len__(self):
            return len(self._v)

        def __repr__(self):
            return f'Ch({self._v})'

    chumpy_mod    = types.ModuleType('chumpy')
    chumpy_ch_mod = types.ModuleType('chumpy.ch')
    chumpy_mod.Ch    = _Ch
    chumpy_ch_mod.Ch = _Ch
    chumpy_mod.array  = np.array
    chumpy_mod.zeros  = np.zeros
    chumpy_mod.ones   = np.ones
    chumpy_mod.arange = np.arange

    sys.modules['chumpy']    = chumpy_mod
    sys.modules['chumpy.ch'] = chumpy_ch_mod


def _to_numpy(v):
    """Convert any chumpy / array-like to a plain numpy array."""
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v
    if hasattr(v, 'r'):          # chumpy.Ch
        return np.array(v.r)
    if hasattr(v, '__array__'):
        return np.array(v)
    return v


def _load_smpl_pkl(pkl_path: str) -> dict:
    """
    Load SMPL pickle file, converting all chumpy objects to numpy arrays.
    """
    _inject_chumpy_stub()

    with open(pkl_path, 'rb') as f:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dd = pickle.load(f, encoding='latin1')

    clean = {}
    for k, v in dd.items():
        try:
            clean[k] = _to_numpy(v)
        except Exception:
            clean[k] = v
    return clean


# ---------------------------------------------------------------------------
# Rodrigues rotation (batch)
# ---------------------------------------------------------------------------

def _rodrigues_batch(rvecs: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of axis-angle vectors to rotation matrices.

    Args:
        rvecs: (N, 3)
    Returns:
        R: (N, 3, 3)
    """
    N = rvecs.shape[0]
    theta = rvecs.norm(dim=1, keepdim=True).clamp(min=1e-8)   # (N, 1)
    r = rvecs / theta                                           # (N, 3)

    cos_t = torch.cos(theta).unsqueeze(-1)                     # (N, 1, 1)
    sin_t = torch.sin(theta).unsqueeze(-1)

    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    zeros = torch.zeros_like(rx)

    K = torch.stack([
        zeros, -rz,  ry,
        rz,  zeros, -rx,
        -ry,  rx,  zeros,
    ], dim=1).reshape(N, 3, 3)                                  # skew-symmetric

    I = torch.eye(3, device=rvecs.device, dtype=rvecs.dtype).unsqueeze(0)
    R = cos_t * I + (1 - cos_t) * r.unsqueeze(-1) * r.unsqueeze(-2) + sin_t * K
    return R


# ---------------------------------------------------------------------------
# SMPLHandler
# ---------------------------------------------------------------------------

class SMPLHandler:
    """
    Pure-PyTorch SMPL forward pass.  No smplx / chumpy dependency.

    Args:
        model_path: Path to SMPL_NEUTRAL.pkl
        device:     torch device
    """

    NUM_JOINTS  = 24
    NUM_BETAS   = 10
    NUM_VERTS   = 6890

    def __init__(self, model_path: str, device: torch.device):
        self.device = device

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SMPL model not found at '{model_path}'. "
                "Download SMPL_NEUTRAL.pkl and place it at that path."
            )

        logger.info(f"Loading SMPL model from {model_path} …")
        dd = _load_smpl_pkl(model_path)

        def _arr(key):
            """Load a value from the pkl and convert to plain float32 numpy array."""
            v = dd.get(key)
            if v is None:
                raise KeyError(f"Key '{key}' not found in SMPL pkl. Available: {list(dd.keys())}")
            # Sparse matrix (scipy) → dense
            if hasattr(v, 'toarray'):
                return v.toarray().astype(np.float32)
            if hasattr(v, 'todense'):
                return np.asarray(v.todense(), dtype=np.float32)
            # chumpy Ch → numpy via .r
            if hasattr(v, 'r'):
                return np.array(v.r, dtype=np.float32)
            return np.array(v, dtype=np.float32)

        def t(key):
            return torch.tensor(_arr(key), dtype=torch.float32, device=device)

        # SMPL parameters — shapes are what the pkl actually contains
        self.v_template  = t('v_template')                                # (6890, 3)
        self.weights     = t('weights')                                    # (6890, 24)

        # shapedirs: pkl has (6890, 3, 300) — use only first NUM_BETAS=10
        sd_raw = _arr('shapedirs')                                         # (6890, 3, 300)
        sd_raw = sd_raw[:, :, :self.NUM_BETAS]                            # (6890, 3, 10)
        self.shapedirs = torch.tensor(sd_raw, dtype=torch.float32, device=device)

        # posedirs: pkl has (6890, 3, 207) — reshape to (6890*3, 207)
        pd_raw = _arr('posedirs')                                          # (6890, 3, 207)
        pd_raw = pd_raw.reshape(-1, pd_raw.shape[-1])                     # (6890*3, 207)
        self.posedirs = torch.tensor(pd_raw, dtype=torch.float32, device=device)

        # J_regressor: may be sparse — convert to dense (24, 6890)
        jr_raw = _arr('J_regressor')
        self.J_regressor = torch.tensor(jr_raw, dtype=torch.float32, device=device)

        faces_np = np.array(dd['f'], dtype=np.int64)
        self._faces = torch.tensor(faces_np, dtype=torch.long, device=device)

        # Kinematic tree — parents[i] = parent joint of joint i
        kintree = np.array(dd['kintree_table'], dtype=np.int64)
        self.parents = torch.tensor(kintree[0], dtype=torch.long, device=device)

        logger.info(
            f"SMPLHandler ready: {self.NUM_VERTS} verts, "
            f"{self._faces.shape[0]} faces, device={device}"
        )

    # ------------------------------------------------------------------

    def get_smpl_mesh(
        self,
        shape_params: torch.Tensor,
        pose_params: torch.Tensor,
    ) -> Meshes:
        """
        SMPL forward pass.

        Args:
            shape_params: (B, 10)
            pose_params:  (B, 72)  — first 3 = global orient, rest = body pose

        Returns:
            Meshes with B meshes of 6890 vertices each.
        """
        B = shape_params.shape[0]
        shape_params = shape_params.to(self.device)
        pose_params  = pose_params.to(self.device)

        with torch.no_grad():
            verts_list = []
            for b in range(B):
                verts = self._forward_single(shape_params[b], pose_params[b])
                verts_list.append(verts)

        faces_exp = self._faces.unsqueeze(0).expand(B, -1, -1)
        return Meshes(verts=verts_list, faces=list(faces_exp))

    def _forward_single(self, betas: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        SMPL forward pass for a single example.

        Args:
            betas: (10,)
            pose:  (72,)  global_orient(3) + body_pose(69)

        Returns:
            vertices: (6890, 3)
        """
        # 1. Shape blend shapes
        #    shapedirs: (6890, 3, 10)  →  einsum over betas
        v_shaped = self.v_template + torch.einsum('ijk,k->ij', self.shapedirs, betas)

        # 2. Joints from shape
        J = self.J_regressor @ v_shaped                        # (24, 3)

        # 3. Pose blend shapes
        #    All 24 rotation matrices (including global orient)
        rvecs = pose.reshape(self.NUM_JOINTS, 3)               # (24, 3)
        R_all = _rodrigues_batch(rvecs)                        # (24, 3, 3)

        #    Pose feature: R[1:] - I  → (23, 3, 3) → flatten to (207,)
        I33 = torch.eye(3, device=self.device, dtype=pose.dtype)
        pose_feat = (R_all[1:] - I33).reshape(-1)              # (207,)
        v_posed = v_shaped + (self.posedirs @ pose_feat).reshape(self.NUM_VERTS, 3)

        # 4. Global transform in kinematic chain
        #    Build joint transforms: T[i] = global transform of joint i
        T_list = [None] * self.NUM_JOINTS
        for i in range(self.NUM_JOINTS):
            local_R = R_all[i]                                 # (3, 3)
            local_t = J[i]                                     # (3,)

            if i == 0:
                parent_t = torch.zeros(3, device=self.device, dtype=pose.dtype)
            else:
                p = self.parents[i].item()
                parent_t = T_list[p][:3, 3]
                local_t = local_t - (self.J_regressor[p] @ v_shaped)

            T_i = torch.eye(4, device=self.device, dtype=pose.dtype)
            T_i[:3, :3] = local_R
            T_i[:3, 3]  = local_t

            if i == 0:
                T_list[i] = T_i
            else:
                T_list[i] = T_list[self.parents[i].item()] @ T_i

        T = torch.stack(T_list, dim=0)                         # (24, 4, 4)

        # 5. Linear Blend Skinning
        #    weights: (6890, 24)
        #    Weighted sum of transforms per vertex
        v_h = torch.cat([v_posed, torch.ones(self.NUM_VERTS, 1, device=self.device, dtype=pose.dtype)], dim=1)
        # (6890, 4)  @  sum_j{ w_j * T_j }  →  (6890, 4)
        W = self.weights                                        # (6890, 24)
        T_w = torch.einsum('vj,jab->vab', W, T)                # (6890, 4, 4)
        v_out = torch.einsum('vab,vb->va', T_w, v_h)           # (6890, 4)

        return v_out[:, :3]                                     # (6890, 3)
