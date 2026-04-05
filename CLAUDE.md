# CLAUDE.md

## Project Overview

**3D Virtual Try-On Backend API** — A FastAPI REST API that takes multi-view photos of a person (front, side, back) plus a garment image and returns a textured 3D GLB model of the garment fitted to a body mesh.

The pipeline uses the SMPL body model to generate a standard T-pose mesh, removes the garment background, then projects the garment texture onto the mesh via orthographic UV mapping. No trained neural-network checkpoints are required for inference.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn (ASGI) |
| Deep Learning | PyTorch + TorchVision |
| 3D Body Model | SMPL (pure-PyTorch LBS, no smplx/chumpy required) |
| 3D Export | trimesh (GLB with PBRMaterial) |
| Image Processing | OpenCV, Pillow |
| Job Queue | Redis (localhost:6379) |
| Frontend | Vanilla HTML/CSS/JS + `<model-viewer>` for GLB preview |

---

## Repository Structure

```
3D_virtual-tryon-backend/
├── run.py                          # Entry point — python run.py (port 8000)
├── app/
│   └── main.py                     # FastAPI app + lifespan startup
├── routes/
│   └── tryon_routes.py             # POST /tryon, GET /result/{job_id}
├── models/
│   ├── garment_classifier.py       # GarmentClassifier (ResNet-18 based)
│   ├── identity_encoder.py         # IdentityEncoder (ResNet-18 → 512-D)
│   ├── vton_model.py               # BodyReconstructionModel, ConditionalGarmentDrapingModel
│   └── smpl/
│       └── SMPL_NEUTRAL.pkl        # SMPL body model (235 MB, gitignored)
├── utils/
│   ├── smpl_handler.py             # Pure-PyTorch SMPL forward pass (LBS)
│   ├── garment_processor.py        # Background removal (GrabCut + white threshold)
│   ├── texture_projector.py        # Orthographic UV projection onto mesh
│   ├── mesh_types.py               # Meshes shim (replaces pytorch3d)
│   ├── job_queue.py                # Redis job queue wrapper
│   ├── template_manager.py         # Garment template management
│   ├── texture_manager.py          # UV texture atlas management
│   ├── texture_warp.py             # Texture warping engine
│   ├── face_extractor.py           # MediaPipe face landmark extraction
│   ├── face_blender.py             # Laplacian pyramid face blending
│   └── preprocessing.py            # Image preprocessing helpers
├── chumpy/
│   ├── __init__.py                 # Minimal chumpy stub (numpy-based)
│   └── ch.py                       # Re-exports from __init__
├── static/
│   └── index.html                  # Frontend UI
├── assets/
│   ├── female_front.jpeg           # Reference test images
│   ├── female_side.jpeg
│   ├── female_back.jpeg
│   └── female_top.jpeg
├── output/                         # Generated GLB files (served at /output)
└── requirements.txt
```

---

## Running the Server

```bash
# Redis must be running first:
redis-server &

# Start the server:
python3 run.py              # http://localhost:8000
python3 run.py --port 9000  # custom port
python3 run.py --reload     # dev mode with auto-reload
```

Open `http://localhost:8000` for the UI, `http://localhost:8000/docs` for API docs.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `TEXTURE_QUALITY` | `fast` | `fast` (1024px) or `high` (2048px) |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves frontend (`static/index.html`) |
| `GET` | `/api/health` | JSON subsystem status |
| `GET` | `/weights/status` | Weight file availability |
| `POST` | `/tryon` | Submit try-on job |
| `GET` | `/result/{job_id}` | Poll job status / get result URL |
| `GET` | `/output/{job_id}.glb` | Download generated GLB |

**POST /tryon** accepts multipart form:
- `user_image_front` — front view of person
- `user_image_side` — side view of person
- `user_image_back` — back view of person
- `garment_image` — garment photo

Returns `{ job_id, status: "pending" }`. Poll `/result/{job_id}` until `status == "completed"`, then fetch the `result_url`.

---

## Inference Pipeline

`routes/tryon_routes.py` → `_inference_pipeline()`:

1. **SMPL T-pose mesh** — zero betas + zero pose → 6890 vertices, 13776 faces (no checkpoints needed)
2. **Garment background removal** — white-pixel detection (R,G,B > 230) + GrabCut refinement + morphological cleanup (`utils/garment_processor.py`)
3. **UV projection** — orthographic projection: U = normalised X, V = 1 − normalised Y; front half (Z ≥ median Z) maps to left half of texture atlas, back to right half (`utils/texture_projector.py`)
4. **GLB export** — trimesh with PBRMaterial (baseColorTexture, metallicFactor=0, roughnessFactor=0.8, doubleSided=True)

Output: `output/{job_id}.glb` (~300–700 KB)

---

## SMPL Handler (`utils/smpl_handler.py`)

Pure-PyTorch implementation — no smplx or chumpy dependency at runtime:

- **Chumpy stub**: injected into `sys.modules` before `pickle.load` to deserialise old `.pkl` files on Python 3.12+
- **pkl key names**: `v_template`, `shapedirs` (6890,3,300), `posedirs` (6890,3,207), `J_regressor` (sparse→dense), `weights`, `f` (faces), `kintree_table`
- **Forward pass**: shape blend shapes → joint regression → Rodrigues rotation → pose blend shapes → kinematic chain transforms → Linear Blend Skinning

---

## Startup Sequence (`app/main.py`)

Loads in order on server start:
1. SMPL model path check
2. Redis job queue
3. TemplateManager
4. TextureManager + TextureWarpEngine
5. SMPLHandler ← **required** for inference
6. GarmentClassifier (optional, falls back to heuristic)
7. BodyReconstructionModel (no checkpoint = random weights, unused by current pipeline)
8. ConditionalGarmentDrapingModel (same)
9. FaceExtractor (requires `pip install mediapipe`, gracefully disabled if missing)
10. IdentityEncoder (ResNet-18)
11. FaceBlender (Laplacian pyramid)

GPU is auto-detected (`cuda:0` if available, else CPU).

---

## Gitignored Assets

```
models/smpl/SMPL_NEUTRAL.pkl   # 235 MB — download separately
checkpoints/                    # .pth checkpoint files (not required for current pipeline)
output/                         # generated GLB files
```

---

## Known Issues / Notes

- **Python 3.14**: chumpy cannot be installed; the inline `sys.modules` stub in `smpl_handler.py` handles this transparently
- **pytorch3d**: not used; replaced by `utils/mesh_types.py` shim
- **checkpoints/**: directory missing triggers a CRITICAL log warning but does NOT prevent startup or inference — the current pipeline skips the neural draping models entirely
- **mediapipe**: optional; face landmark extraction is disabled gracefully if not installed
- Single-worker only (`--workers 1`). ML models are loaded into `app.state` and are not process-safe across multiple workers.
