"""
Virtual Try-On API Routes

New inference pipeline:
  Input images → PIFuHD → 3D human mesh
                ↓
          OpenPose (keypoints)
                ↓
  Garment image → U²Net segmentation
                ↓
        VITON garment warp (GMM + TPS)
                ↓
  Render warped garment on 3D avatar → GLB
"""

import asyncio
import logging
import os
import uuid

import cv2
import numpy as np
import torch
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from PIL import Image

from utils.job_queue import JobQueue

logger = logging.getLogger(__name__)

router = APIRouter()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Inference pipeline (blocking — runs in thread pool via asyncio.to_thread)
# ─────────────────────────────────────────────────────────────────────────────

def _inference_pipeline(
    front_bytes: bytes,
    side_bytes:  bytes,
    back_bytes:  bytes,
    garment_bytes: bytes,
    job_id: str,
    app,
) -> str:
    """
    Full virtual try-on pipeline:

    1. OpenPose  — detect 18 body keypoints from the front image
    2. PIFuHD    — reconstruct a 3-D body mesh from the front image
    3. U²Net     — segment the garment (foreground extraction)
    4. VITON     — warp the garment to body shape (GMM + TPS)
    5. Render    — project warped garment texture onto the 3-D mesh → GLB
    """
    try:
        # ── retrieve models from app state ────────────────────────────────────
        openpose_handler  = getattr(app.state, "openpose_handler",  None)
        pifuhd_handler    = getattr(app.state, "pifuhd_handler",    None)
        u2net_handler     = getattr(app.state, "u2net_handler",     None)
        viton_warper      = getattr(app.state, "viton_warper",      None)

        if pifuhd_handler is None:
            raise ValueError("PIFuHDHandler not initialised")

        # ── 1. OpenPose — body keypoints ──────────────────────────────────────
        logger.info(f"[{job_id}] Step 1/5 — OpenPose keypoint detection …")
        if openpose_handler is not None:
            keypoints = openpose_handler.detect(front_bytes)
            logger.info(f"[{job_id}] Keypoints detected (conf mean: "
                        f"{keypoints.confidence.mean():.2f})")
        else:
            from utils.openpose_handler import OpenPoseHandler, Keypoints
            logger.warning(f"[{job_id}] OpenPoseHandler not in app.state — "
                           "creating ad-hoc (synthetic fallback)")
            openpose_handler = OpenPoseHandler(DEVICE)
            keypoints = openpose_handler.detect(front_bytes)

        # ── 2. PIFuHD — 3-D mesh reconstruction ──────────────────────────────
        logger.info(f"[{job_id}] Step 2/5 — PIFuHD 3-D reconstruction …")
        verts, faces = pifuhd_handler.reconstruct(front_bytes, keypoints=keypoints)
        logger.info(f"[{job_id}] Mesh: {verts.shape[0]} verts, {faces.shape[0]} faces")

        # ── 3. U²Net — garment segmentation ──────────────────────────────────
        logger.info(f"[{job_id}] Step 3/5 — U²Net garment segmentation …")
        if u2net_handler is not None:
            garment_texture = u2net_handler.get_garment_texture(garment_bytes, size=1024)
            mask, garment_rgba = u2net_handler.segment(garment_bytes)
        else:
            from utils.garment_processor import prepare_garment_texture
            logger.warning(f"[{job_id}] U2NetHandler not in app.state — using GrabCut fallback")
            garment_texture = prepare_garment_texture(garment_bytes, size=1024)
            # Build dummy rgba
            mask = np.ones(garment_texture.shape[:2], dtype=np.float32)
            alpha = (mask * 255).astype(np.uint8)
            garment_rgba = np.dstack([garment_texture, alpha])

        logger.info(f"[{job_id}] Garment mask coverage: "
                    f"{mask.mean() * 100:.1f}%")

        # ── 4. VITON — garment warping ────────────────────────────────────────
        logger.info(f"[{job_id}] Step 4/5 — VITON garment warp …")
        front_img = np.array(Image.open(__import__("io").BytesIO(front_bytes)).convert("RGB"))

        from utils.viton_warper import build_person_repr
        person_repr = build_person_repr(
            front_img,
            keypoints,
            body_mask=None,
            output_size=256,
        )

        if viton_warper is not None:
            warped_garment = viton_warper.warp(garment_rgba, person_repr, keypoints)
        else:
            from utils.viton_warper import VITONWarper
            logger.warning(f"[{job_id}] VITONWarper not in app.state — creating ad-hoc")
            viton_warper_local = VITONWarper(DEVICE)
            warped_garment = viton_warper_local.warp(garment_rgba, person_repr, keypoints)

        warped_rgb  = warped_garment[:, :, :3]   # (H, W, 3)
        warped_mask = warped_garment[:, :, 3]    # (H, W)  uint8 [0,255]

        logger.info(f"[{job_id}] Warped garment: {warped_rgb.shape}")

        # ── 5. Render on 3-D avatar ───────────────────────────────────────────
        logger.info(f"[{job_id}] Step 5/5 — Rendering warped garment on 3-D avatar …")

        # Resize warped garment to texture atlas size
        tex_size = 1024
        warped_tex  = cv2.resize(warped_rgb,  (tex_size, tex_size), interpolation=cv2.INTER_LINEAR)
        warped_mask_r = cv2.resize(warped_mask, (tex_size, tex_size), interpolation=cv2.INTER_LINEAR)

        # Blend warped garment over the base garment texture using mask
        garment_base = cv2.resize(garment_texture, (tex_size, tex_size))
        alpha_f = (warped_mask_r / 255.0)[:, :, np.newaxis]
        blended_tex = (warped_tex.astype(np.float32) * alpha_f
                       + garment_base.astype(np.float32) * (1 - alpha_f))
        blended_tex = np.clip(blended_tex, 0, 255).astype(np.uint8)

        # Generate UV coordinates for the mesh
        from utils.texture_projector import project_garment_onto_mesh
        uv_coords, final_texture = project_garment_onto_mesh(verts, faces, blended_tex)

        # Export textured GLB
        import trimesh
        import trimesh.visual
        import trimesh.visual.material

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        pil_tex = Image.fromarray(final_texture.astype(np.uint8), 'RGB').convert('RGBA')
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=pil_tex,
            metallicFactor=0.0,
            roughnessFactor=0.8,
            doubleSided=True,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv_coords, material=material)

        output_path = f"output/{job_id}.glb"
        os.makedirs("output", exist_ok=True)
        glb_data = mesh.export(file_type='glb')
        with open(output_path, 'wb') as fh:
            fh.write(glb_data)

        file_size = len(glb_data)
        logger.info(f"[{job_id}] GLB written to {output_path} ({file_size:,} bytes)")

        if file_size < 10_000:
            logger.warning(f"[{job_id}] GLB is unusually small ({file_size} bytes)")

        return f"/{output_path}"

    except Exception as exc:
        logger.error(f"[{job_id}] Pipeline error: {exc}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Async wrapper
# ─────────────────────────────────────────────────────────────────────────────

async def run_tryon_pipeline(
    job_id: str,
    job_queue: JobQueue,
    front_bytes:   bytes,
    side_bytes:    bytes,
    back_bytes:    bytes,
    garment_bytes: bytes,
    app,
) -> None:
    try:
        await asyncio.to_thread(job_queue.update_job_status, job_id, status="processing")

        output_url = await asyncio.to_thread(
            _inference_pipeline,
            front_bytes, side_bytes, back_bytes, garment_bytes, job_id, app,
        )

        await asyncio.to_thread(
            job_queue.update_job_status, job_id, status="completed", result_url=output_url
        )
    except Exception as exc:
        logger.error(f"[{job_id}] Background task failed: {exc}", exc_info=True)
        await asyncio.to_thread(
            job_queue.update_job_status, job_id, status="failed", error=str(exc)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/tryon", tags=["Try-On"])
async def start_tryon(
    request: Request,
    background_tasks: BackgroundTasks,
    user_image_front: UploadFile = File(..., description="Front view of the user"),
    user_image_side:  UploadFile = File(..., description="Side view of the user"),
    user_image_back:  UploadFile = File(..., description="Back view of the user"),
    garment_image:    UploadFile = File(..., description="Garment image"),
):
    """
    Submit a new virtual try-on job.

    Pipeline: OpenPose → PIFuHD → U²Net → VITON → 3-D avatar render

    Returns a job_id to poll via GET /result/{job_id}.
    """
    job_queue: JobQueue = request.app.state.job_queue
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue (Redis) unavailable.")

    job_id = str(uuid.uuid4())
    if not await asyncio.to_thread(job_queue.create_job, job_id):
        raise HTTPException(status_code=500, detail="Failed to create job.")

    try:
        front_bytes   = await user_image_front.read()
        side_bytes    = await user_image_side.read()
        back_bytes    = await user_image_back.read()
        garment_bytes = await garment_image.read()

        if not all([front_bytes, side_bytes, back_bytes, garment_bytes]):
            raise HTTPException(status_code=400, detail="One or more images are empty.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error reading images: {exc}")

    background_tasks.add_task(
        run_tryon_pipeline,
        job_id, job_queue,
        front_bytes, side_bytes, back_bytes, garment_bytes,
        request.app,
    )

    return {
        "job_id":  job_id,
        "status":  "pending",
        "message": "Try-on job submitted. Poll /result/{job_id} for status.",
        "pipeline": ["OpenPose", "PIFuHD", "U2Net", "VITON", "3D-Render"],
    }


@router.get("/result/{job_id}", tags=["Try-On"])
async def get_result(job_id: str, request: Request):
    """
    Poll for job status or retrieve the result URL when complete.

    Status values: pending | processing | completed | failed
    """
    job_queue: JobQueue = request.app.state.job_queue
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue unavailable.")

    status_data = await asyncio.to_thread(job_queue.get_job_status, job_id)
    if status_data is None:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    return status_data
