"""
mesh_utils.py — 3D Mesh Texture Utilities for Virtual Try-On
==============================================================

Provides functions to create UV-textured trimesh objects and export
them to .glb format. Fixes the blank-white-mesh issue by properly
attaching TextureVisuals with PBRMaterial and UV coordinates.

Usage:
    from mesh_utils import create_textured_mesh, export_mesh_to_glb

    mesh = create_textured_mesh(vertices, faces, uv_coords, texture_img)
    export_mesh_to_glb(mesh, "output.glb")
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import trimesh
import trimesh.visual
import trimesh.visual.material
from PIL import Image

logger = logging.getLogger(__name__)


def create_textured_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray,
    texture_image: Union[str, Path, Image.Image],
    metallic_factor: float = 0.0,
    roughness_factor: float = 0.8,
) -> trimesh.Trimesh:
    """
    Create a trimesh.Trimesh with proper UV-mapped texture for .glb export.

    This function fixes the common issue where trimesh defaults to
    ColorVisuals (solid white) when no texture is explicitly attached.

    Args:
        vertices:  (N, 3) float array — vertex positions.
        faces:     (F, 3) int array   — triangle face indices.
        uv_coords: (N, 2) float array — per-vertex UV coordinates in [0, 1].
        texture_image: PIL Image, or path to the texture file.
        metallic_factor:  PBR metallic  (0.0 = dielectric, 1.0 = metal).
        roughness_factor: PBR roughness (0.0 = mirror, 1.0 = diffuse).

    Returns:
        trimesh.Trimesh with TextureVisuals applied, ready for .glb export.

    Raises:
        ValueError: If array shapes are inconsistent or UV coords out of range.
    """
    # ── Validate inputs ──────────────────────────────────────────
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    uv_coords = np.asarray(uv_coords, dtype=np.float64)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3), got {faces.shape}")
    if uv_coords.ndim != 2 or uv_coords.shape[1] != 2:
        raise ValueError(f"uv_coords must be (N, 2), got {uv_coords.shape}")
    if uv_coords.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"uv_coords length ({uv_coords.shape[0]}) must match "
            f"vertices length ({vertices.shape[0]})"
        )

    # Clamp UVs to [0, 1] to avoid texture sampling artefacts
    if uv_coords.min() < 0.0 or uv_coords.max() > 1.0:
        logger.warning("UV coordinates outside [0, 1] — clamping.")
        uv_coords = np.clip(uv_coords, 0.0, 1.0)

    # ── Load texture image ───────────────────────────────────────
    if isinstance(texture_image, (str, Path)):
        texture_image = Image.open(texture_image)
    texture_image = texture_image.convert("RGBA")

    logger.info(
        "Creating textured mesh: %d verts, %d faces, texture %s",
        len(vertices),
        len(faces),
        texture_image.size,
    )

    # ── Build PBR Material ───────────────────────────────────────
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture_image,
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=metallic_factor,
        roughnessFactor=roughness_factor,
        doubleSided=True,
    )

    # ── Build TextureVisuals ─────────────────────────────────────
    #  This is the critical fix: trimesh requires an explicit
    #  TextureVisuals object with UV coords + material to embed
    #  textures in glTF/GLB exports. Without this, it falls back
    #  to a plain white ColorVisuals.
    texture_visuals = trimesh.visual.TextureVisuals(
        uv=uv_coords,
        material=material,
    )

    # ── Construct Mesh ───────────────────────────────────────────
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=texture_visuals,
        process=False,  # IMPORTANT: prevent vertex reordering that
                        # would break the UV <-> vertex mapping.
    )

    # Sanity check
    assert isinstance(
        mesh.visual, trimesh.visual.TextureVisuals
    ), f"Expected TextureVisuals, got {type(mesh.visual).__name__}"

    return mesh


def export_mesh_to_glb(
    mesh: trimesh.Trimesh,
    output_path: Union[str, Path],
) -> Path:
    """
    Export a textured Trimesh to a binary glTF (.glb) file.

    Args:
        mesh:        A trimesh.Trimesh (ideally with TextureVisuals).
        output_path: Destination .glb file path.

    Returns:
        pathlib.Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    glb_data = mesh.export(file_type="glb")
    output_path.write_bytes(glb_data)

    size_kb = len(glb_data) / 1024
    logger.info("Exported .glb → %s (%.1f KB)", output_path, size_kb)

    if size_kb < 5:
        logger.warning(
            "GLB file is very small (%.1f KB) — texture may not have "
            "been embedded correctly.",
            size_kb,
        )

    return output_path


def create_body_mesh_with_clothing_texture(
    body_vertices: np.ndarray,
    body_faces: np.ndarray,
    body_uv: np.ndarray,
    clothing_texture_path: Union[str, Path, Image.Image],
    output_glb_path: Union[str, Path],
) -> Path:
    """
    High-level convenience: build a SMPL/SMPL-X body mesh with a
    clothing texture overlay and export to .glb.

    This is the function you should call from your FastAPI endpoint
    to replace the current broken export logic.

    Args:
        body_vertices:       (N, 3) SMPL(-X) body vertices.
        body_faces:          (F, 3) face indices.
        body_uv:             (N, 2) UV coordinates from the SMPL UV map.
        clothing_texture_path: Path to the 2D clothing texture image,
                               OR a PIL.Image.Image directly.
        output_glb_path:     Where to save the .glb file.

    Returns:
        Path to the exported .glb file.
    """
    if isinstance(clothing_texture_path, Image.Image):
        texture_img = clothing_texture_path.convert("RGBA")
    else:
        texture_img = Image.open(clothing_texture_path).convert("RGBA")

    mesh = create_textured_mesh(
        vertices=body_vertices,
        faces=body_faces,
        uv_coords=body_uv,
        texture_image=texture_img,
    )

    return export_mesh_to_glb(mesh, output_glb_path)


# ── SMPL UV Coordinate Helpers ───────────────────────────────────

def load_smpl_uv(smpl_uv_path: Union[str, Path]) -> np.ndarray:
    """
    Load SMPL/SMPL-X UV coordinates from a .npy or .obj file.

    Args:
        smpl_uv_path: Path to the UV data file.

    Returns:
        (N, 2) numpy array of UV coordinates.
    """
    path = Path(smpl_uv_path)

    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".obj":
        return _parse_uv_from_obj(path)
    else:
        raise ValueError(f"Unsupported UV file format: {path.suffix}")


def _parse_uv_from_obj(obj_path: Path) -> np.ndarray:
    """Parse 'vt' lines from a Wavefront OBJ file."""
    uvs = []
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("vt "):
                parts = line.strip().split()
                uvs.append([float(parts[1]), float(parts[2])])
    return np.array(uvs, dtype=np.float64)
