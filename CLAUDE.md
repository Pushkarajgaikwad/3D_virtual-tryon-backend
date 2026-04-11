# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**3D Virtual Try-On Backend API** — A FastAPI REST API that takes multi-view photos of a person (front, side, back) plus a garment image and returns a textured 3D GLB model of the garment fitted to a reconstructed body mesh.

**Current inference pipeline:**
```
Input images → OpenPose (keypoints) → PIFuHD (3D mesh)
Garment image → U²Net (segmentation) → VITON GMM (TPS warp)
Warped garment projected onto 3D mesh → GLB export
```

Every stage has a graceful fallback so the server runs with zero checkpoints.

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

`run.py` kills any existing process on the target port before binding.  
UI: `http://localhost:8000` | API docs: `http://localhost:8000/docs`

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `TEXTURE_QUALITY` | `fast` | `fast` (1024 px) or `high` (2048 px) |

---

## Inference Pipeline

`routes/tryon_routes.py` → `_inference_pipeline()` runs five sequential steps:

1. **OpenPose** (`utils/openpose_handler.py`) — detects 18 body keypoints from the front image. Fallback chain: CNN checkpoint → MediaPipe Pose → synthetic T-pose.

2. **PIFuHD** (`utils/pifuhd_handler.py`) — reconstructs a 3-D body mesh via pixel-aligned implicit functions (stacked hourglass encoder + occupancy MLP + marching cubes). Fallback: SMPL neutral mesh (zero betas/pose, 6890 verts / 13776 faces).

3. **U²Net** (`utils/u2net_handler.py`) — segments the garment with a nested U²-structure CNN (RSU blocks). Outputs a float mask and an RGBA cut-out. Fallback: GrabCut + white-threshold (`utils/garment_processor.py`).

4. **VITON GMM** (`utils/viton_warper.py`) — warps the segmented garment to fit the body using a Geometric Matching Module (VGG correlation → TPS control-point regression). Fallback: keypoint-guided `cv2.getPerspectiveTransform`.

5. **Render + export** — warped garment texture is blended onto a base texture, UV-projected onto the mesh via `utils/texture_projector.py`, and exported as a GLB with `trimesh` PBRMaterial.

Output: `output/{job_id}.glb` served at `/output/{job_id}.glb`.

---

## Checkpoint Paths

Place trained weights here (all optional — fallbacks activate when absent):

| File | Stage | Notes |
|---|---|---|
| `checkpoints/openpose_body.pth` | OpenPose CNN | VGG-19 + PAF, 18 keypoints |
| `checkpoints/pifuhd_final.pt` | PIFuHD | Requires `scikit-image` for marching cubes |
| `checkpoints/u2net.pth` | U²Net | ImageNet-normalised input (320×320) |
| `checkpoints/gmm.pth` | VITON GMM | TPS warp, 5×5 control grid |
| `models/smpl/SMPL_NEUTRAL.pkl` | SMPL (PIFuHD fallback) | 235 MB, gitignored |
| `checkpoints/vton_body_model.pth` | BodyReconstructionModel | Legacy, unused by active pipeline |
| `checkpoints/vton_garment_model_conditional.pth` | GarmentDrapingModel | Legacy, unused |

---

## Startup Sequence (`app/main.py`)

Models are loaded into `app.state` in this order:

1. Weight-file existence check (non-blocking)
2. Redis `JobQueue`
3. `TemplateManager`, `TextureManager`, `TextureWarpEngine`
4. `SMPLHandler` (SMPL_NEUTRAL.pkl — required for PIFuHD fallback)
5. `GarmentClassifier`, `BodyReconstructionModel`, `ConditionalGarmentDrapingModel` (legacy)
6. **`OpenPoseHandler`** — new pipeline
7. **`PIFuHDHandler`** — new pipeline (receives `smpl_handler` as fallback)
8. **`U2NetHandler`** — new pipeline
9. **`VITONWarper`** — new pipeline
10. `FaceExtractor`, `IdentityEncoder`, `FaceBlender` (legacy face components)

GPU is auto-detected (`cuda:0` if available, else CPU). Single-worker only — `app.state` is not process-safe across multiple uvicorn workers.

---

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Frontend (`static/index.html`) |
| `GET` | `/api/health` | Subsystem status — includes `pipeline` block for new components |
| `GET` | `/weights/status` | Weight file availability |
| `POST` | `/tryon` | Submit job (multipart: `user_image_front`, `user_image_side`, `user_image_back`, `garment_image`) |
| `GET` | `/result/{job_id}` | Poll: `pending` → `processing` → `completed`/`failed` |
| `GET` | `/output/{job_id}.glb` | Download result |

---

## Key Architectural Notes

**SMPL / chumpy compatibility** — `utils/smpl_handler.py` injects a minimal chumpy stub into `sys.modules` before `pickle.load` so the `.pkl` file deserialises on Python 3.12+. No `smplx` or `chumpy` package needed at runtime.

**pytorch3d not used** — `utils/mesh_types.py` provides a lightweight `Meshes` shim (verts/faces lists, `join_meshes_as_scene`) that replaces the pytorch3d dependency.

**Person representation for VITON** — `viton_warper.build_person_repr()` produces a 22-channel tensor: 3 (RGB) + 1 (body silhouette) + 18 (Gaussian pose heatmaps, one per OpenPose joint). This is the input to the GMM alongside the garment image.

**Texture atlas layout** — `texture_projector.project_garment_onto_mesh()` classifies SMPL/PIFuHD vertices into SHIRT (y ∈ [0.48, 0.82], not hands) and SKIN regions, then builds a 3-section atlas: `[garment_front | garment_back_mirror | skin_tone]`.

**Job queue** — Redis keys hold JSON with `status`, `result_url`, and `error`. TTL = 3600 s. The blocking `_inference_pipeline()` runs via `asyncio.to_thread` so it does not block the event loop.

---

## Known Issues / Notes

- `checkpoints/` directory missing triggers a CRITICAL log warning but does **not** prevent startup — all four pipeline stages fall back gracefully.
- `mediapipe` is optional; install it (`pip install mediapipe`) to activate the OpenPose MediaPipe fallback.
- `scikit-image` is required for PIFuHD marching cubes — without it, PIFuHD falls back to SMPL even if a checkpoint is present.
- The `build_person_repr` body-mask heuristic (centre 60%×90% of image) is used when no explicit silhouette is provided. Replace with a real parser model for better warp quality.
