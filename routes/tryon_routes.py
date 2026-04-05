"""
Virtual Try-On API Routes

Integrates all ML models and utilities into REST endpoints.
"""

import asyncio
import logging
import os
import uuid

import numpy as np
import torch
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from PIL import Image

from utils.job_queue import JobQueue
from utils.smpl_handler import SMPLHandler

logger = logging.getLogger(__name__)

router = APIRouter()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Inference pipeline (blocking — runs in thread pool)
# ---------------------------------------------------------------------------

def _inference_pipeline(
    front_bytes: bytes,
    side_bytes: bytes,
    back_bytes: bytes,
    garment_bytes: bytes,
    job_id: str,
    app,
) -> str:
    """
    Virtual try-on pipeline using SMPL body mesh + garment UV projection.

    No trained neural-network checkpoints required.  Works entirely from:
      • SMPL_NEUTRAL.pkl  (body shape model)
      • The uploaded garment image

    Steps
    -----
    1. Generate a neutral T-pose SMPL body mesh (zero betas → average body).
    2. Remove the garment image background (GrabCut + white-threshold).
    3. Project the garment texture onto the mesh via orthographic UV mapping.
    4. Export a textured GLB file.
    """
    try:
        smpl_handler: SMPLHandler = app.state.smpl_handler

        if smpl_handler is None:
            raise ValueError("SMPLHandler not initialised — SMPL_NEUTRAL.pkl missing")

        # ------------------------------------------------------------------ #
        # 1. Generate body mesh (default shape, T-pose — no checkpoints needed)
        # ------------------------------------------------------------------ #
        logger.info(f"[{job_id}] Generating SMPL T-pose body mesh …")
        shape_params = torch.zeros(1, 10)
        pose_params  = torch.zeros(1, 72)
        body_mesh = smpl_handler.get_smpl_mesh(shape_params, pose_params)

        # ------------------------------------------------------------------ #
        # 2. Extract vertices / faces as numpy arrays
        # ------------------------------------------------------------------ #
        verts = body_mesh.verts_list()[0].cpu().numpy()   # (6890, 3)
        faces = body_mesh.faces_list()[0].cpu().numpy()   # (13776, 3)
        logger.info(f"[{job_id}] Mesh: {verts.shape[0]} verts, {faces.shape[0]} faces")

        # ------------------------------------------------------------------ #
        # 3. Prepare garment texture (background removal + resize)
        # ------------------------------------------------------------------ #
        logger.info(f"[{job_id}] Processing garment image …")
        from utils.garment_processor import prepare_garment_texture
        texture = prepare_garment_texture(garment_bytes, size=1024)   # (1024, 1024, 3)

        # ------------------------------------------------------------------ #
        # 4. Project garment onto mesh → UV coordinates + texture atlas
        # ------------------------------------------------------------------ #
        logger.info(f"[{job_id}] Computing UV projection …")
        from utils.texture_projector import project_garment_onto_mesh
        uv_coords, final_texture = project_garment_onto_mesh(verts, faces, texture)

        # ------------------------------------------------------------------ #
        # 5. Export as textured GLB via trimesh
        # ------------------------------------------------------------------ #
        logger.info(f"[{job_id}] Exporting textured GLB …")
        import trimesh
        import trimesh.visual
        import trimesh.visual.material

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        pil_texture = Image.fromarray(final_texture.astype(np.uint8), 'RGB').convert('RGBA')
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=pil_texture,
            metallicFactor=0.0,
            roughnessFactor=0.8,
            doubleSided=True,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv_coords, material=material)

        output_path = f"output/{job_id}.glb"
        os.makedirs("output", exist_ok=True)
        glb_data = mesh.export(file_type='glb')
        with open(output_path, 'wb') as f:
            f.write(glb_data)

        file_size = len(glb_data)
        logger.info(f"[{job_id}] GLB written to {output_path} ({file_size:,} bytes)")

        if file_size < 10_000:
            logger.warning(f"[{job_id}] GLB is unusually small ({file_size} bytes)")

        return f"/{output_path}"

    except Exception as e:
        logger.error(f"[{job_id}] Pipeline error: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------

async def run_tryon_pipeline(
    job_id: str,
    job_queue: JobQueue,
    front_bytes: bytes,
    side_bytes: bytes,
    back_bytes: bytes,
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
    except Exception as e:
        logger.error(f"[{job_id}] Background task failed: {e}", exc_info=True)
        await asyncio.to_thread(
            job_queue.update_job_status, job_id, status="failed", error=str(e)
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/tryon", tags=["Try-On"])
async def start_tryon(
    request: Request,
    background_tasks: BackgroundTasks,
    user_image_front: UploadFile = File(..., description="Front view of the user"),
    user_image_side: UploadFile = File(..., description="Side view of the user"),
    user_image_back: UploadFile = File(..., description="Back view of the user"),
    garment_image: UploadFile = File(..., description="Garment image"),
):
    """
    Submit a new virtual try-on job.

    Returns a job_id to poll via GET /result/{job_id}.
    """
    job_queue: JobQueue = request.app.state.job_queue
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue (Redis) unavailable.")

    job_id = str(uuid.uuid4())
    if not await asyncio.to_thread(job_queue.create_job, job_id):
        raise HTTPException(status_code=500, detail="Failed to create job.")

    try:
        front_bytes = await user_image_front.read()
        side_bytes = await user_image_side.read()
        back_bytes = await user_image_back.read()
        garment_bytes = await garment_image.read()

        if not all([front_bytes, side_bytes, back_bytes, garment_bytes]):
            raise HTTPException(status_code=400, detail="One or more images are empty.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading images: {e}")

    background_tasks.add_task(
        run_tryon_pipeline,
        job_id, job_queue,
        front_bytes, side_bytes, back_bytes, garment_bytes,
        request.app,
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Try-on job submitted. Poll /result/{job_id} for status.",
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
