"""
Virtual Try-On API Routes (Part 4: Full Integration)

Integrates:
- TemplateManager (template selection and loading)
- GarmentClassifier (template category prediction)
- ConditionalGarmentDrapingModel (FiLM-conditioned GNN)
- BodyReconstructionModel (SMPL parameter regression)
- SMPLHandler (SMPL mesh generation)
- JobQueue (Redis-based async job management)
"""

import torch
import asyncio
import uuid
import logging
from fastapi import (
    APIRouter,
    Request,
    BackgroundTasks,
    UploadFile,
    File,
    HTTPException
)
from pytorch3d.structures import Meshes

# --- Internal Imports ---
from utils.job_queue import JobQueue
from utils.preprocessing import preprocess_person_image, preprocess_garment_image
from utils.mesh_exporter import export_3d_model
from utils.smpl_handler import SMPLHandler
from utils.template_manager import TemplateManager
from utils.texture_manager import TextureManager
from utils.texture_warp import TextureWarpEngine
from utils.face_extractor import FaceExtractor
from utils.face_blender import FaceBlender
from models.vton_model import ConditionalGarmentDrapingModel, BodyReconstructionModel
from models.garment_classifier import GarmentClassifier
from models.identity_encoder import IdentityEncoder
from PIL import Image

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- Constants ---
router = APIRouter()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMB_DIM = 16




def _inference_pipeline(front_bytes: bytes,
                       side_bytes: bytes,
                       back_bytes: bytes,
                       garment_bytes: bytes,
                       job_id: str,
                       app) -> str:
    """
    The blocking AI inference pipeline with template management + texture + identity preservation.
    
    Workflow:
    1. Preprocess 3-view person images + garment image
    2. Run BodyReconstructionModel on all 3 views, average predictions
    3. Generate SMPL body mesh via SMPLHandler
    4. Classify/select garment template via GarmentClassifier or TemplateManager
    5. Load template and get template embedding
    6. Run ConditionalGarmentDrapingModel with SMPL params + template embedding
    7. TEXTURE PIPELINE:
       a. Generate texture from garment image (segment + extract + flatten)
       b. Generate UV map for template
       c. Warp texture to deformed mesh geometry
       d. Generate vertex colors for realistic rendering
    8. FACIAL IDENTITY PIPELINE (NEW):
       a. Extract face landmarks from front view
       b. Extract and align face patch
       c. Encode face identity (512-D embedding)
       d. Convert 3D render to 2D image
       e. Blend face onto final 2D output using Laplacian pyramid
    9. Export combined body+garment+texture mesh with blended face
    
    Args:
        front_bytes: Raw bytes of front view image
        side_bytes: Raw bytes of side view image
        back_bytes: Raw bytes of back view image
        garment_bytes: Raw bytes of garment image
        job_id: Unique job identifier
        app: FastAPI app instance (for accessing app.state)
    
    Returns:
        URL path to the saved 3D model file (.glb)
    
    Raises:
        ValueError: If required models/managers not initialized
        RuntimeError: If inference fails
    """
    try:
        # --- Extract dependencies from app.state ---
        template_manager = app.state.template_manager
        garment_classifier = app.state.garment_classifier
        smpl_handler = app.state.smpl_handler
        body_model = app.state.body_model
        draping_model = app.state.draping_model
        texture_manager = app.state.texture_manager
        texture_warp_engine = app.state.texture_warp_engine
        texture_quality = app.state.texture_quality
        face_extractor = app.state.face_extractor
        identity_encoder = app.state.identity_encoder
        face_blender = app.state.face_blender
        
        if not all([template_manager, smpl_handler, body_model, draping_model]):
            raise ValueError(
                "Required models/managers not initialized. "
                "Check startup logs and server configuration."
            )
        
        # --- 1. Preprocess Images ---
        logger.info(f"[{job_id}] Preprocessing input images (3 views + garment)...")
        
        front_tensor = preprocess_person_image(front_bytes).to(DEVICE)
        side_tensor = preprocess_person_image(side_bytes).to(DEVICE)
        back_tensor = preprocess_person_image(back_bytes).to(DEVICE)
        person_tensor = torch.cat([front_tensor, side_tensor, back_tensor], dim=0)
        
        garment_tensor, garment_mask = preprocess_garment_image(garment_bytes)
        logger.info(f"[{job_id}] Images preprocessed: person {person_tensor.shape}, garment {garment_tensor.shape}")
        
        # --- 2. Stage 1: Body Reconstruction (Multi-View) ---
        logger.info(f"[{job_id}] Running Stage 1: Body Reconstruction (3 views)...")
        with torch.no_grad():
            shape_params_list = []
            pose_params_list = []
            
            for view_idx, view_name in enumerate(['front', 'side', 'back']):
                view_tensor = person_tensor[view_idx:view_idx+1]  # [1, 3, 224, 224]
                shape, pose = body_model(view_tensor)
                shape_params_list.append(shape)
                pose_params_list.append(pose)
            
            # Average across views for robust estimate
            shape_params = torch.mean(torch.cat(shape_params_list, dim=0), dim=0, keepdim=True)
            pose_params = torch.mean(torch.cat(pose_params_list, dim=0), dim=0, keepdim=True)
            
            logger.info(f"[{job_id}] Body parameters averaged across 3 views")
        
        # --- 3. Generate SMPL Body Mesh ---
        logger.info(f"[{job_id}] Generating SMPL body mesh...")
        body_mesh = smpl_handler.get_smpl_mesh(shape_params, pose_params)
        
        # --- 4. Template Selection & Loading ---
        logger.info(f"[{job_id}] Selecting garment template...")
        
        # Try to use GarmentClassifier if available
        template_id = None
        if garment_classifier:
            try:
                template_id, confidence = garment_classifier.predict_category(
                    garment_bytes, template_manager, device=DEVICE
                )
                logger.info(f"[{job_id}] Classifier predicted template '{template_id}' (confidence: {confidence:.3f})")
            except Exception as e:
                logger.warning(f"[{job_id}] Classifier failed, falling back to mask analysis: {e}")
                template_id = None
        
        # Fallback: use TemplateManager heuristic
        if not template_id:
            try:
                template_id = template_manager.select_template_by_image(garment_bytes)
                logger.info(f"[{job_id}] Selected template '{template_id}' via mask analysis")
            except Exception as e:
                # Last resort: use first available template
                available = template_manager.list_templates()
                if available:
                    template_id = available[0]
                    logger.warning(f"[{job_id}] Using default template '{template_id}'")
                else:
                    raise ValueError("No templates available in TemplateManager")
        
        # Load template data
        template_dict = template_manager.load_template(template_id)
        template_verts = template_dict['vertices'].to(DEVICE)
        template_edges = template_dict['edges'].to(DEVICE)
        template_faces = template_dict['faces']
        
        # Get template embedding
        template_emb = template_manager.get_template_embedding(template_id, EMB_DIM).unsqueeze(0)
        template_emb = template_emb.to(DEVICE)
        
        logger.info(f"[{job_id}] Template '{template_id}' loaded: {template_verts.shape[0]} verts")
        
        # --- 5. Stage 2: Conditional Garment Draping ---
        logger.info(f"[{job_id}] Running Stage 2: Garment Draping (with template conditioning)...")
        
        # Combine SMPL parameters
        smpl_params = torch.cat([shape_params, pose_params], dim=1).to(DEVICE)  # [1, 82]
        
        # Batch index: all nodes belong to batch 0
        batch_index = torch.zeros(template_verts.shape[0], dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            offsets = draping_model(
                node_features=template_verts,
                edge_index=template_edges,
                smpl_params=smpl_params,
                template_emb=template_emb,
                batch_index=batch_index
            )  # [N, 3]
            
            logger.info(f"[{job_id}] Offsets computed: shape {offsets.shape}")
        
        # Apply offsets to template to get draped garment
        draped_verts = template_verts + offsets
        
        # Create PyTorch3D mesh
        garment_mesh = Meshes(
            verts=[draped_verts.cpu()],
            faces=[template_faces.cpu()] if template_faces is not None else [torch.tensor([])]
        )
        
        # --- 6. TEXTURE PIPELINE ---
        logger.info(f"[{job_id}] Starting texture pipeline (quality={texture_quality})...")
        
        vertex_colors = None
        warped_texture = None
        
        if texture_manager and texture_warp_engine:
            try:
                # Step 6a: Generate texture from garment image
                logger.info(f"[{job_id}] Generating texture (extract + segment + flatten)...")
                texture, mask = texture_manager.generate_texture(
                    garment_bytes, template_id, quality=texture_quality
                )
                logger.info(f"[{job_id}] Texture generated: {texture.shape}")
                
                # Step 6b: Load UV map for template
                logger.info(f"[{job_id}] Loading UV map for template '{template_id}'...")
                uv_map = texture_manager.load_template_uv(template_id, mesh=template_verts.cpu().numpy())
                logger.info(f"[{job_id}] UV map loaded: {uv_map.shape}")
                
                # Step 6c: Warp texture to deformed mesh
                logger.info(f"[{job_id}] Warping texture to deformed mesh...")
                vertex_colors, warped_texture = texture_warp_engine.warp_texture_to_mesh(
                    texture,
                    uv_map,
                    draped_verts.cpu().numpy(),
                    template_faces.numpy() if isinstance(template_faces, torch.Tensor) else template_faces,
                    quality=texture_quality
                )
                logger.info(f"[{job_id}] Vertex colors generated: {vertex_colors.shape}")
                
            except Exception as e:
                logger.warning(f"[{job_id}] Texture pipeline failed, continuing without textures: {e}")
                vertex_colors = None
                warped_texture = None
        else:
            logger.warning(f"[{job_id}] TextureManager or TextureWarpEngine not available, skipping texture")
        
        # --- 7. Export 3D Model (with optional textures) ---
        logger.info(f"[{job_id}] Exporting final 3D model (with textures)...")
        output_path = f"output/{job_id}"
        
        export_3d_model(
            body_mesh=body_mesh.cpu(),
            garment_mesh=garment_mesh.cpu(),
            output_path=output_path,
            vertex_colors=vertex_colors,
            texture_image=warped_texture,
            texture_quality=texture_quality
        )
        
        # --- 8. FACIAL IDENTITY BLENDING (NEW) ---
        logger.info(f"[{job_id}] Starting facial identity blending pipeline...")
        
        final_image_path = f"{output_path}_final.png"
        identity_blend_failed = False
        
        if face_extractor and identity_encoder and face_blender:
            try:
                # Step 8a: Extract face from front view
                logger.info(f"[{job_id}] Extracting face from front view...")
                try:
                    face_landmarks = face_extractor.detect_landmarks(front_bytes)
                    face_patch, face_landmarks_norm, face_bbox = face_extractor.extract_face_patch(
                        front_bytes, padding=0.2
                    )
                    logger.info(f"[{job_id}] Face extracted: bbox {face_bbox}, {len(face_landmarks)} landmarks")
                except Exception as e:
                    logger.warning(f"[{job_id}] Face extraction failed: {e}. Skipping identity blending.")
                    face_patch = None
                    identity_blend_failed = True
                
                # Step 8b: Encode face identity (if extraction succeeded)
                if face_patch and not identity_blend_failed:
                    try:
                        logger.info(f"[{job_id}] Encoding face identity...")
                        face_identity_emb = identity_encoder.extract_identity(face_patch)
                        logger.info(f"[{job_id}] Face identity encoded: {face_identity_emb.shape}")
                    except Exception as e:
                        logger.warning(f"[{job_id}] Identity encoding failed: {e}. Skipping blending.")
                        identity_blend_failed = True
                
                # Step 8c: Convert 3D render to 2D image (placeholder)
                if not identity_blend_failed:
                    try:
                        logger.info(f"[{job_id}] Converting 3D render to 2D (placeholder: using skin-tone background)...")
                        # Placeholder: Use skin-tone background
                        # In production: render textured mesh to 2D image
                        rendered_width, rendered_height = 512, 640
                        rendered_image = Image.new('RGB', (rendered_width, rendered_height), color=(220, 200, 180))
                        logger.info(f"[{job_id}] 2D render placeholder created: {rendered_image.size}")
                        
                        # Step 8d: Blend face onto rendered output
                        logger.info(f"[{job_id}] Blending face onto final output...")
                        face_bbox_scaled = (
                            int(face_bbox[0] * rendered_width / 512),
                            int(face_bbox[1] * rendered_height / 640),
                            int(face_bbox[2] * rendered_width / 512),
                            int(face_bbox[3] * rendered_height / 640)
                        )
                        
                        final_image = face_blender.blend_face_with_identity(
                            rendered_image,
                            face_patch,
                            face_bbox_scaled,
                            landmarks=face_landmarks,
                            identity_strength=1.0
                        )
                        logger.info(f"[{job_id}] Face blended successfully: {final_image.size}")
                        
                        # Save final blended image
                        final_image.save(final_image_path)
                        logger.info(f"[{job_id}] Final image saved to {final_image_path}")
                    
                    except Exception as e:
                        logger.warning(f"[{job_id}] Face blending failed: {e}. Skipping final blend.")
                        identity_blend_failed = True
            
            except Exception as e:
                logger.warning(f"[{job_id}] Unexpected error in identity pipeline: {e}")
                identity_blend_failed = True
        else:
            logger.warning(f"[{job_id}] Face extraction, identity encoder, or blender not available. Skipping identity preservation.")
            identity_blend_failed = True
        
        if identity_blend_failed:
            logger.warning(f"[{job_id}] Identity blending disabled or failed. Output is base textured mesh without face.")
        
        logger.info(f"[{job_id}] Pipeline complete. Output saved to {output_path}")
        return f"/{output_path}.glb"
        
    except Exception as e:
        logger.error(f"[{job_id}] ERROR in inference pipeline: {e}", exc_info=True)
        raise




async def run_tryon_pipeline(job_id: str,
                            job_queue: JobQueue,
                            front_bytes: bytes,
                            side_bytes: bytes,
                            back_bytes: bytes,
                            garment_bytes: bytes,
                            app) -> None:
    """
    Async wrapper for the inference pipeline.
    Manages job status and runs blocking ML code in thread pool.
    
    Args:
        job_id: Unique job identifier
        job_queue: Redis job queue instance
        front_bytes: Front view image bytes
        side_bytes: Side view image bytes
        back_bytes: Back view image bytes
        garment_bytes: Garment image bytes
        app: FastAPI app instance
    """
    try:
        # Mark as processing
        await asyncio.to_thread(
            job_queue.update_job_status, job_id, status='processing'
        )
        
        # Run inference in thread pool
        output_url = await asyncio.to_thread(
            _inference_pipeline,
            front_bytes,
            side_bytes,
            back_bytes,
            garment_bytes,
            job_id,
            app
        )
        
        # Mark as completed
        await asyncio.to_thread(
            job_queue.update_job_status,
            job_id,
            status='completed',
            result_url=output_url
        )
    
    except Exception as e:
        logger.error(f"[{job_id}] Pipeline failed: {e}", exc_info=True)
        await asyncio.to_thread(
            job_queue.update_job_status,
            job_id,
            status='failed',
            error=str(e)
        )



# --- API Endpoints ---

@router.post("/tryon", tags=["Try-On"])
async def start_tryon(
    request: Request,
    background_tasks: BackgroundTasks,
    user_image_front: UploadFile = File(..., description="Front view image of the user (COMPULSORY)."),
    user_image_side: UploadFile = File(..., description="Side view image of the user (COMPULSORY)."),
    user_image_back: UploadFile = File(..., description="Back view image of the user (COMPULSORY)."),
    garment_image: UploadFile = File(..., description="Image of the garment (COMPULSORY).")
):
    """
    Start a new Virtual Try-On job.
    
    Accepts THREE user images (front, side, back views) and a garment image.
    Multi-view approach enables robust 3D body reconstruction.
    
    Returns:
    - job_id: Use this to poll /result/{job_id}
    - status: 'pending' (processing will run in background)
    """
    job_queue: JobQueue = request.app.state.job_queue
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue (Redis) unavailable.")
    
    # Create job entry
    job_id = str(uuid.uuid4())
    if not await asyncio.to_thread(job_queue.create_job, job_id):
        raise HTTPException(status_code=500, detail="Failed to create job in queue.")
    
    # Read image bytes
    try:
        front_bytes = await user_image_front.read()
        if not front_bytes:
            raise HTTPException(status_code=400, detail="Front view image is empty.")
        
        side_bytes = await user_image_side.read()
        if not side_bytes:
            raise HTTPException(status_code=400, detail="Side view image is empty.")
        
        back_bytes = await user_image_back.read()
        if not back_bytes:
            raise HTTPException(status_code=400, detail="Back view image is empty.")
        
        garment_bytes = await garment_image.read()
        if not garment_bytes:
            raise HTTPException(status_code=400, detail="Garment image is empty.")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading images: {str(e)}")
    
    # Schedule background task (pass app instance)
    background_tasks.add_task(
        run_tryon_pipeline,
        job_id,
        job_queue,
        front_bytes,
        side_bytes,
        back_bytes,
        garment_bytes,
        request.app
    )
    
    return {
        "job_id": job_id,
        "status": "pending",
        "message": "Try-on job submitted. Poll /result/{job_id} for status."
    }


@router.get("/result/{job_id}", tags=["Try-On"])
async def get_result(job_id: str, request: Request):
    """
    Poll endpoint to get job status or result.
    
    Returns:
    - status: 'pending', 'processing', 'completed', or 'failed'
    - result_url: (if completed) URL to .glb file
    - error: (if failed) Error message
    """
    job_queue: JobQueue = request.app.state.job_queue
    if not job_queue:
        raise HTTPException(status_code=503, detail="Job queue unavailable.")
    
    status_data = await asyncio.to_thread(job_queue.get_job_status, job_id)
    
    if not status_data:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    
    return status_data