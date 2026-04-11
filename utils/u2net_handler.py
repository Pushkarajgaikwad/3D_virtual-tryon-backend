"""
U²Net Handler — Deep salient-object segmentation for garment extraction.

Implements U²Net (U-squared Net) — a nested U-structure with Residual
U-blocks (RSU) at multiple scales — for precise garment segmentation.

The segmentation mask is used instead of the white-threshold / GrabCut
approach, producing cleaner garment cutouts particularly for non-white
backgrounds.

Fallback
--------
If the U²Net checkpoint (checkpoints/u2net.pth) is not present, the
handler falls back to the GrabCut + luminance-threshold approach from
garment_processor.py.

Reference
---------
Qin et al., "U²-Net: Going Deeper with Nested U-Structure for Salient
Object Detection", Pattern Recognition, 2020.
"""

import io
import logging
import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

U2NET_CKPT = "checkpoints/u2net.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class RSU(nn.Module):
    """
    Recurrent Residual U-block — the core building block of U²Net.
    Encodes context at multiple scales within a single block.

    Parameters
    ----------
    height   : depth of the inner U-structure (number of encoder stages)
    in_ch    : input channels
    mid_ch   : intermediate channels
    out_ch   : output channels
    """

    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.height = height

        # Input projection
        self.inp = _ConvBNReLU(in_ch, out_ch)

        # Encoder
        self.enc = nn.ModuleList()
        self.enc.append(_ConvBNReLU(out_ch, mid_ch))
        for i in range(1, height - 1):
            self.enc.append(_ConvBNReLU(mid_ch, mid_ch))
        # Bottleneck (dilated)
        self.enc.append(_ConvBNReLU(mid_ch, mid_ch, dilation=2))

        # Decoder
        self.dec = nn.ModuleList()
        for _ in range(height - 1):
            self.dec.append(_ConvBNReLU(mid_ch * 2, mid_ch))
        self.dec_out = _ConvBNReLU(mid_ch * 2, out_ch)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = self.inp(x)

        # Encode
        enc_outs = []
        feat = hx
        for i, layer in enumerate(self.enc):
            if i < len(self.enc) - 1:
                feat = layer(feat)
                enc_outs.append(feat)
                feat = self.pool(feat)
            else:
                feat = layer(feat)   # bottleneck — no pool

        # Decode
        for i, layer in enumerate(self.dec):
            skip = enc_outs[-(i + 1)]
            feat = F.interpolate(feat, size=skip.shape[2:], mode='bilinear', align_corners=False)
            feat = layer(torch.cat([feat, skip], dim=1))

        # Final skip from input
        feat = F.interpolate(feat, size=hx.shape[2:], mode='bilinear', align_corners=False)
        feat = self.dec_out(torch.cat([feat, enc_outs[0] if enc_outs else feat], dim=1))

        return feat + hx   # residual


class RSU4F(nn.Module):
    """RSU with dilated convolutions (no spatial downsampling) for deepest stage."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.inp  = _ConvBNReLU(in_ch,   out_ch)
        self.d1   = _ConvBNReLU(out_ch,  mid_ch, dilation=1)
        self.d2   = _ConvBNReLU(mid_ch,  mid_ch, dilation=2)
        self.d3   = _ConvBNReLU(mid_ch,  mid_ch, dilation=4)
        self.d4   = _ConvBNReLU(mid_ch,  mid_ch, dilation=8)
        self.u3   = _ConvBNReLU(mid_ch * 2, mid_ch, dilation=4)
        self.u2   = _ConvBNReLU(mid_ch * 2, mid_ch, dilation=2)
        self.u1   = _ConvBNReLU(mid_ch * 2, out_ch, dilation=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hx = self.inp(x)
        d1 = self.d1(hx)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        u3 = self.u3(torch.cat([d4, d3], 1))
        u2 = self.u2(torch.cat([u3, d2], 1))
        u1 = self.u1(torch.cat([u2, d1], 1))
        return u1 + hx


# ─────────────────────────────────────────────────────────────────────────────
# U²Net architecture (full model)
# ─────────────────────────────────────────────────────────────────────────────

class U2Net(nn.Module):
    """
    U²Net (full version).
    Input : (B, 3, 320, 320)  normalised to [0,1]
    Output: (B, 1, 320, 320)  saliency map in [0,1] (after sigmoid)
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.stage1  = RSU(7, 3,   32,  64)
        self.stage2  = RSU(6, 64,  32, 128)
        self.stage3  = RSU(5, 128, 64, 256)
        self.stage4  = RSU(4, 256, 128, 512)
        self.stage5  = RSU4F(512, 256, 512)
        self.stage6  = RSU4F(512, 256, 512)

        # Decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU(4, 1024, 128, 256)
        self.stage3d = RSU(5, 512,   64, 128)
        self.stage2d = RSU(6, 256,   32,  64)
        self.stage1d = RSU(7, 128,   16,  64)

        # Side outputs (deep supervision)
        self.side1 = nn.Conv2d(64,  1, 3, padding=1)
        self.side2 = nn.Conv2d(64,  1, 3, padding=1)
        self.side3 = nn.Conv2d(128, 1, 3, padding=1)
        self.side4 = nn.Conv2d(256, 1, 3, padding=1)
        self.side5 = nn.Conv2d(512, 1, 3, padding=1)
        self.side6 = nn.Conv2d(512, 1, 3, padding=1)
        self.fuse  = nn.Conv2d(6,   1, 1)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]

        # Encoder
        h1 = self.stage1(x)
        h2 = self.stage2(self.pool(h1))
        h3 = self.stage3(self.pool(h2))
        h4 = self.stage4(self.pool(h3))
        h5 = self.stage5(self.pool(h4))
        h6 = self.stage6(self.pool(h5))

        # Decoder
        h5d = self.stage5d(torch.cat([
            F.interpolate(h6, size=h5.shape[2:], mode='bilinear', align_corners=False),
            h5
        ], 1))
        h4d = self.stage4d(torch.cat([
            F.interpolate(h5d, size=h4.shape[2:], mode='bilinear', align_corners=False),
            h4
        ], 1))
        h3d = self.stage3d(torch.cat([
            F.interpolate(h4d, size=h3.shape[2:], mode='bilinear', align_corners=False),
            h3
        ], 1))
        h2d = self.stage2d(torch.cat([
            F.interpolate(h3d, size=h2.shape[2:], mode='bilinear', align_corners=False),
            h2
        ], 1))
        h1d = self.stage1d(torch.cat([
            F.interpolate(h2d, size=h1.shape[2:], mode='bilinear', align_corners=False),
            h1
        ], 1))

        # Side outputs → fuse
        def _side(layer, feat):
            return F.interpolate(layer(feat), size=(H, W), mode='bilinear', align_corners=False)

        s1 = _side(self.side1, h1d)
        s2 = _side(self.side2, h2d)
        s3 = _side(self.side3, h3d)
        s4 = _side(self.side4, h4d)
        s5 = _side(self.side5, h5d)
        s6 = _side(self.side6, h6)
        fuse = self.fuse(torch.cat([s1, s2, s3, s4, s5, s6], 1))

        return torch.sigmoid(fuse)   # (B, 1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Public handler
# ─────────────────────────────────────────────────────────────────────────────

class U2NetHandler:
    """
    Segments garment images using U²Net.

    Usage
    -----
    handler = U2NetHandler(device)
    mask, rgba = handler.segment(image_bytes)
      mask : (H, W) float32 in [0,1]
      rgba : (H, W, 4) uint8 with alpha = mask
    """

    INPUT_SIZE = 320  # U²Net canonical input

    def __init__(self, device: torch.device, checkpoint_path: str = U2NET_CKPT):
        self.device = device
        self._net: Optional[U2Net] = None
        self._init(checkpoint_path)

    # ── init ─────────────────────────────────────────────────────────────────

    def _init(self, ckpt_path: str):
        if os.path.exists(ckpt_path):
            try:
                net = U2Net().to(self.device)
                state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                net.load_state_dict(state)
                net.eval()
                self._net = net
                logger.info(f"✓ U2NetHandler: checkpoint loaded from {ckpt_path}")
                return
            except Exception as exc:
                logger.warning(f"⚠ U2NetHandler: checkpoint load failed ({exc}) — using GrabCut fallback")

        logger.info("✓ U2NetHandler: GrabCut fallback active")

    # ── public API ────────────────────────────────────────────────────────────

    def segment(self, image_bytes: bytes) -> tuple:
        """
        Returns
        -------
        mask : (H, W) float32 in [0,1]   — garment foreground probability
        rgba : (H, W, 4) uint8            — RGBA image with alpha = mask×255
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if self._net is not None:
            mask = self._segment_u2net(img)
        else:
            mask = self._segment_grabcut(img)

        # Build RGBA output
        rgb = np.array(img, dtype=np.uint8)
        alpha = (mask * 255).astype(np.uint8)
        rgba = np.dstack([rgb, alpha])
        return mask, rgba

    def get_garment_texture(self, image_bytes: bytes, size: int = 1024) -> np.ndarray:
        """
        Segment garment and return a clean RGB texture (H×H×3) with
        background filled by average garment colour.
        """
        img_orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask, _ = self.segment(image_bytes)

        # Resize both mask and image to square texture
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), 'L').resize(
            (size, size), Image.BILINEAR)
        img_sq = img_orig.resize((size, size), Image.LANCZOS)

        mask_arr = np.array(mask_pil) / 255.0   # (size, size) [0,1]
        rgb_arr  = np.array(img_sq, dtype=np.float32)

        # Average garment colour (foreground pixels)
        fg_mask = mask_arr > 0.5
        if fg_mask.any():
            avg_color = rgb_arr[fg_mask].mean(axis=0)
        else:
            avg_color = np.array([255.0, 255.0, 255.0])

        # Fill background with average colour
        out = rgb_arr.copy()
        bg = ~fg_mask
        out[bg] = avg_color[np.newaxis, :]

        # Soft-blend at edges
        soft_mask = mask_arr[:, :, np.newaxis]
        out = out * soft_mask + avg_color[np.newaxis, np.newaxis, :] * (1 - soft_mask)

        return np.clip(out, 0, 255).astype(np.uint8)

    # ── U²Net inference ───────────────────────────────────────────────────────

    def _segment_u2net(self, img: Image.Image) -> np.ndarray:
        S = self.INPUT_SIZE
        orig_W, orig_H = img.size
        img_r = img.resize((S, S), Image.BILINEAR)

        # Normalise with ImageNet stats (U²Net convention)
        arr = np.array(img_r, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        arr  = (arr - mean) / std

        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            prob = self._net(x)  # (1, 1, S, S)

        mask_s = prob[0, 0].cpu().numpy()   # (S, S) [0,1]
        # Resize back to original
        mask = np.array(Image.fromarray(mask_s).resize((orig_W, orig_H), Image.BILINEAR))
        return mask.astype(np.float32)

    # ── GrabCut fallback ──────────────────────────────────────────────────────

    @staticmethod
    def _segment_grabcut(img: Image.Image) -> np.ndarray:
        arr = np.array(img.convert("RGB"))
        H, W = arr.shape[:2]

        # 1. White-pixel mask (fast threshold)
        is_white = (arr[:, :, 0] > 230) & (arr[:, :, 1] > 230) & (arr[:, :, 2] > 230)
        rough_fg = (~is_white).astype(np.uint8)

        # 2. Largest connected component in rough fg
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(rough_fg, connectivity=8)
        if n_labels > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            cc_mask = (labels == largest).astype(np.uint8) * 255
        else:
            cc_mask = rough_fg * 255

        # 3. GrabCut refinement
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        mask_gc   = np.where(cc_mask > 127,
                             cv2.GC_FGD, cv2.GC_BGD).astype(np.uint8)
        rect = (5, 5, W - 10, H - 10)
        try:
            cv2.grabCut(arr, mask_gc, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            fg = ((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD)).astype(np.float32)
        except Exception:
            fg = (cc_mask > 127).astype(np.float32)

        # 4. Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel)

        return fg.astype(np.float32)
