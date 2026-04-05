"""
Virtual Try-On 3D Backend API (Part 4: Full Integration)

Entry point for FastAPI application with startup/shutdown management.
Initializes all ML models, templates, and queue systems on startup.
"""

from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import os
import logging
import torch

from utils.job_queue import JobQueue
from utils.smpl_handler import SMPLHandler
from utils.template_manager import TemplateManager
from utils.texture_manager import TextureManager
from utils.texture_warp import TextureWarpEngine
from utils.face_extractor import FaceExtractor
from utils.face_blender import FaceBlender
from models.garment_classifier import GarmentClassifier
from models.identity_encoder import IdentityEncoder
from models.vton_model import BodyReconstructionModel, ConditionalGarmentDrapingModel
from routes import tryon_routes

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration Constants ---
OUTPUT_DIR = "output"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

TEMPLATES_ROOT = "templates"
SMPL_MODEL_PATH = "models/smpl/SMPL_NEUTRAL.pkl"
BODY_MODEL_PATH = "checkpoints/vton_body_model.pth"
GARMENT_MODEL_PATH = "checkpoints/vton_garment_model_conditional.pth"
GARMENT_CLASSIFIER_PATH = "models/garment_classifier.pth"

# --- Texture Configuration (NEW) ---
TEXTURE_QUALITY = os.getenv("TEXTURE_QUALITY", "fast")  # "fast" or "high"
TEXTURE_RESOLUTION = 1024 if TEXTURE_QUALITY == "fast" else 2048

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMB_DIM = 16


# --- Application Lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    
    Startup:
    - Initialize Redis job queue
    - Load TemplateManager
    - Load TextureManager + TextureWarpEngine
    - Load SMPLHandler
    - Load GarmentClassifier (optional)
    - Load BodyReconstructionModel and ConditionalGarmentDrapingModel
    - Create output directory
    
    Shutdown:
    - Close Redis connection
    """
    # --- STARTUP ---
    logger.info("="*70)
    logger.info("Application Starting Up")
    logger.info("="*70)
    
    try:
        # 1. Initialize Redis Job Queue
        logger.info(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
        job_queue = JobQueue(host=REDIS_HOST, port=REDIS_PORT)
        app.state.job_queue = job_queue
        logger.info("✓ Redis job queue connected")
    except Exception as e:
        logger.error(f"✗ Failed to connect to Redis: {e}")
        app.state.job_queue = None
    
    try:
        # 2. Initialize TemplateManager
        logger.info(f"Initializing TemplateManager from {TEMPLATES_ROOT}...")
        template_manager = TemplateManager(templates_root=TEMPLATES_ROOT)
        templates = template_manager.list_templates()
        app.state.template_manager = template_manager
        logger.info(f"✓ TemplateManager loaded with templates: {templates}")
    except Exception as e:
        logger.error(f"✗ Failed to initialize TemplateManager: {e}")
        app.state.template_manager = None
    
    try:
        # 3. Initialize TextureManager (NEW)
        logger.info(f"Initializing TextureManager (texture_quality={TEXTURE_QUALITY})...")
        texture_manager = TextureManager(templates_root=TEMPLATES_ROOT, uv_res=TEXTURE_RESOLUTION)
        app.state.texture_manager = texture_manager
        logger.info(f"✓ TextureManager initialized (resolution={TEXTURE_RESOLUTION}px)")
    except Exception as e:
        logger.warning(f"⚠ Failed to initialize TextureManager (textures disabled): {e}")
        app.state.texture_manager = None
    
    try:
        # 4. Initialize TextureWarpEngine (NEW)
        logger.info(f"Initializing TextureWarpEngine...")
        texture_warp_engine = TextureWarpEngine(texture_resolution=TEXTURE_RESOLUTION)
        app.state.texture_warp_engine = texture_warp_engine
        logger.info(f"✓ TextureWarpEngine initialized")
    except Exception as e:
        logger.warning(f"⚠ Failed to initialize TextureWarpEngine (textures disabled): {e}")
        app.state.texture_warp_engine = None
    
    # Store texture quality setting
    app.state.texture_quality = TEXTURE_QUALITY
    logger.info(f"Texture quality set to: {TEXTURE_QUALITY}")
    
    try:
        # 5. Initialize SMPLHandler
        logger.info(f"Initializing SMPLHandler from {SMPL_MODEL_PATH}...")
        smpl_handler = SMPLHandler(model_path=SMPL_MODEL_PATH, device=DEVICE)
        app.state.smpl_handler = smpl_handler
        logger.info("✓ SMPLHandler initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize SMPLHandler: {e}")
        app.state.smpl_handler = None
    
    try:
        # 6. Initialize GarmentClassifier (optional)
        logger.info("Initializing GarmentClassifier...")
        garment_classifier = GarmentClassifier(
            num_classes=len(template_manager.list_templates()) if app.state.template_manager else 2,
            emb_dim=EMB_DIM
        ).to(DEVICE)
        garment_classifier.eval()
        
        # Try to load checkpoint
        if os.path.exists(GARMENT_CLASSIFIER_PATH):
            logger.info(f"Loading classifier checkpoint from {GARMENT_CLASSIFIER_PATH}...")
            garment_classifier.load(GARMENT_CLASSIFIER_PATH, device=DEVICE)
            logger.info("✓ GarmentClassifier loaded with checkpoint")
        else:
            logger.warning(f"✓ GarmentClassifier initialized (no checkpoint found at {GARMENT_CLASSIFIER_PATH})")
        
        app.state.garment_classifier = garment_classifier
    except Exception as e:
        logger.warning(f"✗ Failed to initialize GarmentClassifier (fallback to heuristic): {e}")
        app.state.garment_classifier = None
    
    try:
        # 7. Initialize BodyReconstructionModel
        logger.info(f"Initializing BodyReconstructionModel...")
        body_model = BodyReconstructionModel(n_iter=3).to(DEVICE)
        body_model.eval()
        
        if os.path.exists(BODY_MODEL_PATH):
            logger.info(f"Loading body model checkpoint from {BODY_MODEL_PATH}...")
            state = torch.load(BODY_MODEL_PATH, map_location=DEVICE, weights_only=True)
            body_model.load_state_dict(state)
            logger.info("✓ BodyReconstructionModel loaded with checkpoint")
        else:
            logger.warning(f"⚠ BodyReconstructionModel initialized (no checkpoint found at {BODY_MODEL_PATH})")
        
        app.state.body_model = body_model
    except Exception as e:
        logger.error(f"✗ Failed to initialize BodyReconstructionModel: {e}")
        app.state.body_model = None
    
    try:
        # 8. Initialize ConditionalGarmentDrapingModel
        logger.info("Initializing ConditionalGarmentDrapingModel...")
        draping_model = ConditionalGarmentDrapingModel(
            node_input_dim=3,
            hidden_dim=128,
            cond_dim=82 + EMB_DIM,
            emb_dim=EMB_DIM,
            num_layers=4
        ).to(DEVICE)
        draping_model.eval()
        
        if os.path.exists(GARMENT_MODEL_PATH):
            logger.info(f"Loading draping model checkpoint from {GARMENT_MODEL_PATH}...")
            state = torch.load(GARMENT_MODEL_PATH, map_location=DEVICE, weights_only=True)
            draping_model.load_state_dict(state)
            logger.info("✓ ConditionalGarmentDrapingModel loaded with checkpoint")
        else:
            logger.warning(f"⚠ ConditionalGarmentDrapingModel initialized (no checkpoint found at {GARMENT_MODEL_PATH})")
        
        app.state.draping_model = draping_model
    except Exception as e:
        logger.error(f"✗ Failed to initialize ConditionalGarmentDrapingModel: {e}")
        app.state.draping_model = None
    
    try:
        # 9. Initialize FaceExtractor (NEW)
        logger.info("Initializing FaceExtractor for facial landmark detection...")
        face_extractor = FaceExtractor(model_selection=0)
        app.state.face_extractor = face_extractor
        logger.info("✓ FaceExtractor initialized (MediaPipe FaceMesh)")
    except Exception as e:
        logger.warning(f"⚠ Failed to initialize FaceExtractor: {e}")
        app.state.face_extractor = None
    
    try:
        # 10. Initialize IdentityEncoder (NEW)
        logger.info("Initializing IdentityEncoder for face embeddings...")
        identity_encoder = IdentityEncoder(embedding_dim=512).to(DEVICE)
        identity_encoder.eval()
        app.state.identity_encoder = identity_encoder
        logger.info("✓ IdentityEncoder initialized (ResNet-18 512-D embeddings)")
    except Exception as e:
        logger.warning(f"⚠ Failed to initialize IdentityEncoder: {e}")
        app.state.identity_encoder = None
    
    try:
        # 11. Initialize FaceBlender (NEW)
        logger.info("Initializing FaceBlender for seamless face blending...")
        face_blender = FaceBlender(pyramid_levels=4)
        app.state.face_blender = face_blender
        logger.info("✓ FaceBlender initialized (Laplacian Pyramid blending)")
    except Exception as e:
        logger.warning(f"⚠ Failed to initialize FaceBlender: {e}")
        app.state.face_blender = None
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"✓ Output directory '{OUTPUT_DIR}' ensured")
    
    logger.info("="*70)
    logger.info("Startup Complete - API Ready")
    logger.info("="*70)
    
    yield  # --- APPLICATION IS RUNNING ---
    
    # --- SHUTDOWN ---
    logger.info("Application shutting down...")
    if app.state.job_queue and hasattr(app.state.job_queue, 'r'):
        try:
            app.state.job_queue.r.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.warning(f"Error closing Redis: {e}")
    
    logger.info("Shutdown complete")


# --- FastAPI App Initialization ---

app = FastAPI(
    title="Virtual Try-On 3D API",
    description="3D virtual clothing try-on with multi-view body reconstruction and template-conditioned garment draping",
    version="2.0.0 (Part 4: Full Integration)",
    lifespan=lifespan
)

# Include all routes
app.include_router(tryon_routes.router)


# --- Health Check Endpoint ---

@app.get("/", tags=["Health"])
async def read_root(request: Request):
    """
    Root endpoint for health checks.
    Confirms API is running and displays status of all subsystems.
    """
    return {
        "message": "Virtual Try-On 3D API",
        "version": "2.0.0",
        "texture_quality": getattr(request.app.state, 'texture_quality', 'unknown'),
        "status": {
            "redis": "connected" if request.app.state.job_queue else "disconnected",
            "template_manager": "ready" if request.app.state.template_manager else "unavailable",
            "texture_manager": "ready" if request.app.state.texture_manager else "unavailable",
            "texture_warp_engine": "ready" if request.app.state.texture_warp_engine else "unavailable",
            "smpl_handler": "ready" if request.app.state.smpl_handler else "unavailable",
            "body_model": "ready" if request.app.state.body_model else "unavailable",
            "draping_model": "ready" if request.app.state.draping_model else "unavailable",
            "classifier": "ready" if request.app.state.garment_classifier else "fallback (heuristic)",
            "face_extractor": "ready" if request.app.state.face_extractor else "unavailable",
            "identity_encoder": "ready" if request.app.state.identity_encoder else "unavailable",
            "face_blender": "ready" if request.app.state.face_blender else "unavailable"
        },
        "docs": "/docs"
    }


# --- Running the Server ---
# Run with: uvicorn main:app --reload
# Or: uvicorn main:app --host 0.0.0.0 --port 8000