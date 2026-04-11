"""
OpenPose Handler — Body keypoint detection (18-joint BODY_18 format).

Priority:
  1. OpenPose CNN  (checkpoints/openpose_body.pth — VGG-19 + PAF)
  2. MediaPipe Pose (automatic fallback if no checkpoint)
  3. Synthetic T-pose proportions (last resort)

Outputs normalised [0,1] keypoint coordinates so downstream stages
(PIFuHD crop, VITON warp) are resolution-independent.
"""

import io
import logging
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

OPENPOSE_CKPT = "checkpoints/openpose_body.pth"

# BODY_18 joint names (OpenPose convention)
KEYPOINT_NAMES = [
    "nose",          # 0
    "neck",          # 1
    "right_shoulder", "right_elbow", "right_wrist",   # 2-4
    "left_shoulder",  "left_elbow",  "left_wrist",    # 5-7
    "right_hip", "right_knee", "right_ankle",          # 8-10
    "left_hip",  "left_knee",  "left_ankle",           # 11-13
    "right_eye", "left_eye",                            # 14-15
    "right_ear", "left_ear",                            # 16-17
]

# Skeleton connections for visualisation / downstream use
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (0, 15), (14, 16), (15, 17),
]


@dataclass
class Keypoints:
    """18 body keypoints with confidence scores."""
    points: np.ndarray    # (18, 2) normalised [0,1] (x, y); -1 = undetected
    confidence: np.ndarray  # (18,) score in [0,1]
    image_size: tuple      # (H, W) of source image


# ─────────────────────────────────────────────────────────────────────────────
# OpenPose CNN (VGG-19 backbone + two-stage PAF / heatmap prediction)
# ─────────────────────────────────────────────────────────────────────────────

def _conv_bn_relu(in_ch: int, out_ch: int, k: int = 3, pad: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=pad),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _OpenPoseNet(nn.Module):
    """
    Lightweight OpenPose body estimator.
    Input : (B, 3, 368, 368) RGB image normalised to [0,1]
    Output: heatmaps (B, 18, 46, 46) — one map per keypoint
            pafs     (B, 38, 46, 46) — part affinity fields (19 limbs × 2)
    """

    NUM_KP  = 18
    NUM_PAF = 38

    def __init__(self):
        super().__init__()
        # Backbone: VGG-19 up to conv4_2 → 128-ch feature maps at 1/8 res
        self.backbone = nn.Sequential(
            _conv_bn_relu(3, 64),  _conv_bn_relu(64, 64),   nn.MaxPool2d(2, 2),
            _conv_bn_relu(64, 128), _conv_bn_relu(128, 128), nn.MaxPool2d(2, 2),
            _conv_bn_relu(128, 256), _conv_bn_relu(256, 256),
            _conv_bn_relu(256, 256), _conv_bn_relu(256, 256), nn.MaxPool2d(2, 2),
            _conv_bn_relu(256, 512), _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 256, 1, 0), _conv_bn_relu(256, 128, 1, 0),
        )  # → (B, 128, 46, 46)

        # Stage-1 heads
        self.paf1 = self._make_head(128, self.NUM_PAF)
        self.hm1  = self._make_head(128, self.NUM_KP + 1)   # +1 background

        # Stage-2 heads (concat feat + paf1 + hm1)
        in2 = 128 + self.NUM_PAF + self.NUM_KP + 1
        self.paf2 = self._make_stage2_head(in2, self.NUM_PAF)
        self.hm2  = self._make_stage2_head(in2, self.NUM_KP + 1)

    @staticmethod
    def _make_head(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            _conv_bn_relu(in_ch, 128), _conv_bn_relu(128, 128),
            _conv_bn_relu(128, 128),   _conv_bn_relu(128, 512, 1, 0),
            nn.Conv2d(512, out_ch, 1),
        )

    @staticmethod
    def _make_stage2_head(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            _conv_bn_relu(in_ch, 128, 7, 3), _conv_bn_relu(128, 128, 7, 3),
            _conv_bn_relu(128, 128, 7, 3),   _conv_bn_relu(128, 128, 1, 0),
            nn.Conv2d(128, out_ch, 1),
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        paf1 = self.paf1(feat)
        hm1  = self.hm1(feat)
        cat  = torch.cat([feat, paf1, hm1], dim=1)
        paf2 = self.paf2(cat)
        hm2  = self.hm2(cat)
        return hm2[:, :self.NUM_KP], paf2   # drop background channel


# ─────────────────────────────────────────────────────────────────────────────
# Public handler
# ─────────────────────────────────────────────────────────────────────────────

class OpenPoseHandler:
    """
    Detects 18 body keypoints from a single RGB image.

    Usage
    -----
    handler = OpenPoseHandler(device)
    kp: Keypoints = handler.detect(image_bytes)
    """

    def __init__(self, device: torch.device, checkpoint_path: str = OPENPOSE_CKPT):
        self.device = device
        self._cnn: Optional[_OpenPoseNet] = None
        self._mp  = None
        self._init(checkpoint_path)

    # ── initialisation ────────────────────────────────────────────────────────

    def _init(self, ckpt_path: str):
        if os.path.exists(ckpt_path):
            try:
                net = _OpenPoseNet().to(self.device)
                state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                net.load_state_dict(state)
                net.eval()
                self._cnn = net
                logger.info(f"✓ OpenPoseHandler: checkpoint loaded from {ckpt_path}")
                return
            except Exception as exc:
                logger.warning(f"⚠ OpenPoseHandler: checkpoint load failed ({exc}) — falling back")

        try:
            import mediapipe as mp
            self._mp = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5,
            )
            logger.info("✓ OpenPoseHandler: MediaPipe Pose fallback active")
        except Exception as exc:
            logger.warning(f"⚠ OpenPoseHandler: MediaPipe unavailable ({exc}) — using synthetic fallback")

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, image_bytes: bytes) -> Keypoints:
        """Detect 18 keypoints from raw image bytes."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        H, W = img.size[1], img.size[0]

        if self._cnn is not None:
            return self._detect_cnn(img, H, W)
        if self._mp is not None:
            return self._detect_mediapipe(img, H, W)
        return self._synthetic(H, W)

    def to_pixel_coords(self, kp: Keypoints) -> np.ndarray:
        """Convert normalised Keypoints to pixel (x, y) array. -1 stays -1."""
        H, W = kp.image_size
        px = kp.points.copy()
        mask = px[:, 0] >= 0
        px[mask, 0] *= W
        px[mask, 1] *= H
        return px  # (18, 2)

    # ── CNN inference ─────────────────────────────────────────────────────────

    def _detect_cnn(self, img: Image.Image, H: int, W: int) -> Keypoints:
        SIZE = 368
        img_r = img.resize((SIZE, SIZE), Image.BILINEAR)
        x = torch.from_numpy(np.array(img_r, dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            hm, _ = self._cnn(x)  # (1, 18, 46, 46)

        hm_np = hm[0].cpu().numpy()  # (18, 46, 46)
        GH, GW = hm_np.shape[1], hm_np.shape[2]
        points = np.full((18, 2), -1.0, dtype=np.float32)
        conf   = np.zeros(18, dtype=np.float32)

        for i in range(18):
            flat = int(np.argmax(hm_np[i]))
            py, px = divmod(flat, GW)
            val = float(hm_np[i, py, px])
            if val > 0.05:
                points[i] = [(px + 0.5) / GW, (py + 0.5) / GH]
                conf[i] = min(val, 1.0)

        return Keypoints(points=points, confidence=conf, image_size=(H, W))

    # ── MediaPipe fallback ────────────────────────────────────────────────────

    # MediaPipe landmark index → OpenPose BODY_18 index
    _MP2OP = {
        0: 0,   # nose
        11: 5,  # left shoulder
        12: 2,  # right shoulder
        13: 6,  # left elbow
        14: 3,  # right elbow
        15: 7,  # left wrist
        16: 4,  # right wrist
        23: 11, # left hip
        24: 8,  # right hip
        25: 12, # left knee
        26: 9,  # right knee
        27: 13, # left ankle
        28: 10, # right ankle
        2:  16, # left ear  (mp 7 = left ear, but 2 = left eye inner)
        5:  14, # right eye
        6:  15, # left eye
        7:  17, # left ear
        8:  16, # right ear
    }

    def _detect_mediapipe(self, img: Image.Image, H: int, W: int) -> Keypoints:
        result = self._mp.process(np.array(img))
        if result.pose_landmarks is None:
            return self._synthetic(H, W)

        lms = result.pose_landmarks.landmark
        points = np.full((18, 2), -1.0, dtype=np.float32)
        conf   = np.zeros(18, dtype=np.float32)

        for mp_i, op_i in self._MP2OP.items():
            if mp_i >= len(lms):
                continue
            lm = lms[mp_i]
            points[op_i] = [np.clip(lm.x, 0, 1), np.clip(lm.y, 0, 1)]
            conf[op_i]   = float(lm.visibility)

        # neck = midpoint shoulders
        rs, ls = points[2], points[5]
        if (rs >= 0).all() and (ls >= 0).all():
            points[1] = (rs + ls) * 0.5
            conf[1]   = (conf[2] + conf[5]) * 0.5

        return Keypoints(points=points, confidence=conf, image_size=(H, W))

    # ── Synthetic T-pose fallback ─────────────────────────────────────────────

    @staticmethod
    def _synthetic(H: int, W: int) -> Keypoints:
        pts = np.array([
            [0.50, 0.07],  # 0  nose
            [0.50, 0.14],  # 1  neck
            [0.65, 0.20],  # 2  R shoulder
            [0.75, 0.34],  # 3  R elbow
            [0.82, 0.48],  # 4  R wrist
            [0.35, 0.20],  # 5  L shoulder
            [0.25, 0.34],  # 6  L elbow
            [0.18, 0.48],  # 7  L wrist
            [0.60, 0.52],  # 8  R hip
            [0.60, 0.72],  # 9  R knee
            [0.60, 0.91],  # 10 R ankle
            [0.40, 0.52],  # 11 L hip
            [0.40, 0.72],  # 12 L knee
            [0.40, 0.91],  # 13 L ankle
            [0.53, 0.05],  # 14 R eye
            [0.47, 0.05],  # 15 L eye
            [0.56, 0.07],  # 16 R ear
            [0.44, 0.07],  # 17 L ear
        ], dtype=np.float32)
        return Keypoints(
            points=pts,
            confidence=np.full(18, 0.4, dtype=np.float32),
            image_size=(H, W),
        )
