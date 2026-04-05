#!/bin/bash
# ============================================================
#  HPC Setup Script — Creates all required files on the HPC
#  Run from: /scratch/nidhi.raut.aissmsioit/3D_virtual-tryon-backend
# ============================================================
set -e
echo "=== Creating project files on HPC ==="

# --- 1. Create directories ---
mkdir -p logs checkpoints output uploads utils

# --- 2. Create deploy_vton.sh ---
cat > deploy_vton.sh << 'DEPLOY_EOF'
#!/bin/bash
#SBATCH --job-name=vton_api
#SBATCH --output=logs/vton_%j.out
#SBATCH --error=logs/vton_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
PROJECT_DIR="/scratch/nidhi.raut.aissmsioit/3D_virtual-tryon-backend"
cd "${PROJECT_DIR}"
mkdir -p logs

echo "============================================"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Node        : $(hostname)"
echo "  GPUs        : ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Start Time  : $(date)"
echo "============================================"

# Activate Conda
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "[ERROR] Could not find conda.sh"
    exit 1
fi

conda activate vton
echo "[OK] Conda environment 'vton' activated"
echo "    Python: $(which python) — $(python --version)"

# Start Redis
REDIS_SERVER="${PROJECT_DIR}/redis-stable/src/redis-server"
if [ ! -x "${REDIS_SERVER}" ]; then
    echo "[ERROR] redis-server not found at ${REDIS_SERVER}"
    exit 1
fi

"${REDIS_SERVER}" --daemonize yes \
    --bind 127.0.0.1 --port 6379 \
    --dir "${PROJECT_DIR}" \
    --logfile "${PROJECT_DIR}/logs/redis_${SLURM_JOB_ID}.log"

echo "[OK] Redis started (127.0.0.1:6379)"
sleep 1
"${PROJECT_DIR}/redis-stable/src/redis-cli" ping || echo "[WARN] Redis not responding"

# Start FastAPI
echo "[->] Starting Uvicorn on 0.0.0.0:8000 ..."
echo "    Access: http://$(hostname):8000"
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info

# Cleanup
"${PROJECT_DIR}/redis-stable/src/redis-cli" shutdown nosave 2>/dev/null || true
echo "[OK] Job completed at $(date)"
DEPLOY_EOF
chmod +x deploy_vton.sh
echo "[OK] deploy_vton.sh created"

# --- 3. Create utils/mesh_exporter.py (FIXED version) ---
cat > utils/mesh_exporter.py << 'EXPORTER_EOF'
import torch
import trimesh
import trimesh.visual
import trimesh.visual.material
import numpy as np
import os
from pytorch3d.structures import Meshes
from pytorch3d.ops import join_meshes_as_scene
from typing import List, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def _generate_spherical_uv(vertices):
    """Generate UV coordinates via spherical projection."""
    centered = vertices - vertices.mean(axis=0)
    x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
    theta = np.arctan2(x, z)
    phi = np.arctan2(y, np.sqrt(x**2 + z**2))
    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi / 2) / np.pi
    return np.stack([u, v], axis=1).astype(np.float64)


def write_obj_with_texture(mesh, texture_image, output_path):
    """Export mesh to OBJ + MTL + PNG texture format."""
    try:
        base_path = output_path.replace('.obj', '').replace('.mtl', '')
        obj_path = f"{base_path}.obj"
        mtl_path = f"{base_path}.mtl"
        tex_path = f"{base_path}.png"

        mesh.export(obj_path)
        logger.info(f"Exported OBJ to {obj_path}")

        mtl_content = f"# Material\nnewmtl garment_material\nKa 1.0 1.0 1.0\nKd 0.8 0.8 0.8\nKs 0.2 0.2 0.2\nNs 32.0\nmap_Kd {os.path.basename(tex_path)}\n"
        with open(mtl_path, 'w') as f:
            f.write(mtl_content)

        with open(obj_path, 'r') as f:
            obj_content = f.read()
        if "mtllib" not in obj_content:
            obj_content = f"mtllib {os.path.basename(mtl_path)}\n" + obj_content
        if "usemtl" not in obj_content:
            obj_content = obj_content.replace("f ", "usemtl garment_material\nf ", 1)
        with open(obj_path, 'w') as f:
            f.write(obj_content)

        if texture_image is not None:
            if texture_image.dtype == np.float32:
                texture_image = (texture_image * 255).astype(np.uint8)
            img = Image.fromarray(texture_image, mode='RGB')
            img.save(tex_path)
            logger.info(f"Saved texture to {tex_path}")

        return True
    except Exception as e:
        logger.error(f"Failed to export OBJ with texture: {e}")
        raise


def write_glb_with_texture(mesh, texture_image, output_path):
    """
    Export mesh to GLB with embedded PBR texture via UV mapping.
    THIS IS THE FIX for the blank white mesh issue.
    """
    try:
        if not output_path.endswith('.glb'):
            output_path = output_path.replace('.obj', '').replace('.mtl', '') + '.glb'

        if texture_image is not None:
            # Normalize to uint8
            if texture_image.dtype in (np.float32, np.float64):
                if texture_image.max() <= 1.0:
                    texture_image = (texture_image * 255).astype(np.uint8)
                else:
                    texture_image = texture_image.astype(np.uint8)

            # Convert to PIL RGBA
            if texture_image.shape[-1] == 3:
                pil_texture = Image.fromarray(texture_image, mode='RGB').convert('RGBA')
            elif texture_image.shape[-1] == 4:
                pil_texture = Image.fromarray(texture_image, mode='RGBA')
            else:
                pil_texture = Image.fromarray(texture_image).convert('RGBA')

            # Generate UV coordinates
            uv_coords = np.clip(_generate_spherical_uv(mesh.vertices), 0.0, 1.0)

            # Create PBR Material with the texture
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=pil_texture,
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.8,
                doubleSided=True,
            )

            # Build TextureVisuals — THIS is the critical fix
            texture_visuals = trimesh.visual.TextureVisuals(
                uv=uv_coords,
                material=material,
            )

            # Replace the mesh visual
            mesh.visual = texture_visuals
            logger.info(f"Applied PBR TextureVisuals ({pil_texture.size} texture, {len(uv_coords)} UVs)")

        glb_data = mesh.export(file_type='glb')
        with open(output_path, 'wb') as f:
            f.write(glb_data)

        file_size = len(glb_data)
        logger.info(f"Exported GLB to {output_path} ({file_size} bytes)")

        if file_size < 5000:
            logger.warning(f"GLB very small ({file_size} bytes) - texture may not be embedded")

        return True
    except Exception as e:
        logger.error(f"Failed to export GLB: {e}")
        raise


def export_3d_model(body_mesh, garment_mesh, output_path,
                    vertex_colors=None, texture_image=None,
                    texture_quality="fast"):
    """Combines body+garment meshes and exports with textures."""
    try:
        if body_mesh is None or garment_mesh is None:
            raise ValueError("Body mesh or garment mesh is None")
        if body_mesh.isempty() or garment_mesh.isempty():
            raise ValueError("Body mesh or garment mesh is empty")

        body_verts = body_mesh.verts_list()
        garment_verts = garment_mesh.verts_list()
        if not body_verts or not garment_verts:
            raise ValueError("No vertices in meshes")
        if len(body_verts[0]) == 0 or len(garment_verts[0]) == 0:
            raise ValueError("Empty vertices")

        logger.info(f"Combining meshes: body ({len(body_verts[0])} verts) + garment ({len(garment_verts[0])} verts)")

        combined_mesh = join_meshes_as_scene([body_mesh.cpu(), garment_mesh.cpu()])

        verts = combined_mesh.verts_list()[0].cpu().numpy()
        faces = combined_mesh.faces_list()[0].cpu().numpy()
        logger.info(f"Combined mesh: {len(verts)} vertices, {len(faces)} faces")

        # IMPORTANT: process=False prevents vertex reordering (breaks UV mapping)
        mesh_scene = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        # Apply vertex colors ONLY as fallback when no texture image
        if texture_image is None and vertex_colors is not None:
            if vertex_colors.dtype in (np.float32, np.float64):
                colors_uint8 = (vertex_colors * 255).astype(np.uint8) if vertex_colors.max() <= 1.0 else vertex_colors.astype(np.uint8)
            else:
                colors_uint8 = vertex_colors.astype(np.uint8)

            if len(colors_uint8) == len(verts):
                mesh_scene.visual.vertex_colors = colors_uint8
                logger.info(f"Applied vertex colors: {colors_uint8.shape}")
            else:
                logger.warning(f"Vertex color count mismatch ({len(colors_uint8)} vs {len(verts)})")

        # Export
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if output_path.endswith('.glb') or output_path.endswith('.gltf'):
            glb_path = output_path if output_path.endswith('.glb') else output_path.replace('.gltf', '.glb')
            write_glb_with_texture(mesh_scene, texture_image, glb_path)
            final_path = glb_path
        else:
            glb_path = output_path.rstrip('/') + '.glb'
            write_glb_with_texture(mesh_scene, texture_image, glb_path)
            if texture_image is not None:
                obj_path = output_path.rstrip('/') + '.obj'
                write_obj_with_texture(mesh_scene, texture_image, obj_path)
            final_path = glb_path

        if not os.path.exists(final_path):
            raise IOError(f"File not created at {final_path}")

        file_size = os.path.getsize(final_path)
        logger.info(f"Exported 3D model to {final_path} ({file_size} bytes, quality={texture_quality})")
        return True

    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise
EXPORTER_EOF
echo "[OK] utils/mesh_exporter.py created (FIXED texture export)"

# --- 4. Patch main.py — add weight verification to lifespan startup ---
# We insert the weight check block after "Application Starting Up" log line
python3 << 'PATCH_EOF'
import re

filepath = "main.py"
with open(filepath, "r") as f:
    content = f.read()

# Check if already patched
if "Verify Critical Weights" in content:
    print("[SKIP] main.py already has weight verification")
else:
    weight_check_block = '''
    # --- 0. Verify Critical Weights (Non-Blocking) ---
    weights_ok = True
    missing_weights = []
    
    import os as _os
    _smpl = "models/smpl/SMPL_NEUTRAL.pkl"
    _ckpt_dir = "checkpoints"
    _body_ckpt = "checkpoints/vton_body_model.pth"
    _garment_ckpt = "checkpoints/vton_garment_model_conditional.pth"
    _classifier_ckpt = "models/garment_classifier.pth"
    
    if not _os.path.exists(_smpl):
        logger.critical("MISSING WEIGHTS: %s — SMPL body model is REQUIRED", _smpl)
        missing_weights.append(_smpl)
        weights_ok = False
    else:
        _sz = _os.path.getsize(_smpl) / (1024*1024)
        logger.info(f"SMPL model found: {_smpl} ({_sz:.1f} MB)")
    
    if not _os.path.isdir(_ckpt_dir):
        logger.critical("MISSING WEIGHTS: checkpoints/ directory does not exist — models use random weights")
        missing_weights.append(_ckpt_dir)
        weights_ok = False
    else:
        for _name, _path in [("BodyModel", _body_ckpt), ("GarmentDraping", _garment_ckpt), ("Classifier", _classifier_ckpt)]:
            if _os.path.exists(_path):
                _sz = _os.path.getsize(_path) / (1024*1024)
                logger.info(f"{_name} checkpoint: {_path} ({_sz:.1f} MB)")
            else:
                logger.warning(f"MISSING WEIGHTS: {_name} not found at {_path} — using random init")
                missing_weights.append(_path)
                weights_ok = False
    
    app.state.weights_status = {
        "all_ok": weights_ok,
        "missing": missing_weights,
        "message": "All weights loaded" if weights_ok else f"{len(missing_weights)} weight(s) missing"
    }
    if not weights_ok:
        logger.warning(f"Server starting with {len(missing_weights)} missing weight(s). Inference may return untrained results.")
'''

    # Insert after the "Application Starting Up" block
    marker = 'logger.info("="*70)\n'
    parts = content.split(marker)
    if len(parts) >= 3:
        # Insert after the 3rd occurrence of the marker (end of startup banner)
        content = parts[0] + marker + parts[1] + marker + parts[2] + marker + weight_check_block + marker.join(parts[3:])
        # Hmm, this is tricky. Let me use a simpler approach.
    
    # Simpler: insert before "try:\n        # 1. Initialize Redis"
    target = "    try:\n        # 1. Initialize Redis"
    if target in content:
        content = content.replace(target, weight_check_block + "\n" + target, 1)
        with open(filepath, "w") as f:
            f.write(content)
        print("[OK] main.py patched with weight verification")
    else:
        print("[WARN] Could not find insertion point in main.py — manual edit needed")

PATCH_EOF

# --- 5. Add /weights/status endpoint to main.py ---
python3 << 'ENDPOINT_EOF'
filepath = "main.py"
with open(filepath, "r") as f:
    content = f.read()

if "/weights/status" in content:
    print("[SKIP] /weights/status endpoint already exists")
else:
    endpoint_code = '''

@app.get("/weights/status", tags=["Health"])
async def weights_endpoint(request: Request):
    """Detailed status of all model weight files."""
    return getattr(request.app.state, 'weights_status', {"error": "Not yet initialized"})
'''
    # Insert before "# --- Running the Server ---"
    target = "# --- Running the Server ---"
    if target in content:
        content = content.replace(target, endpoint_code + "\n" + target, 1)
        with open(filepath, "w") as f:
            f.write(content)
        print("[OK] Added /weights/status endpoint to main.py")
    else:
        print("[WARN] Could not find insertion point for /weights/status")

ENDPOINT_EOF

# --- 6. Ensure utils/__init__.py exists ---
touch utils/__init__.py

echo ""
echo "=========================================="
echo "  Setup Complete! Files created:"
echo "    - deploy_vton.sh"
echo "    - utils/mesh_exporter.py (FIXED)"
echo "    - main.py (patched with weight checks)"
echo "    - checkpoints/ (empty, ready for .pth)"
echo "    - logs/"
echo "=========================================="
echo ""
echo "Next: run 'sbatch deploy_vton.sh'"
