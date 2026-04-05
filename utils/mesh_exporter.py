import torch
import trimesh
import trimesh.visual
import trimesh.visual.material
import numpy as np
import os
from utils.mesh_types import Meshes, join_meshes_as_scene
from typing import List, Optional
from PIL import Image
import logging

# Setup logging
logger = logging.getLogger(__name__)


def _generate_spherical_uv(vertices: np.ndarray) -> np.ndarray:
    """
    Generate UV coordinates via spherical projection.
    Fallback when no explicit UV map is available.
    """
    centered = vertices - vertices.mean(axis=0)
    x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]

    theta = np.arctan2(x, z)
    phi = np.arctan2(y, np.sqrt(x**2 + z**2))

    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi / 2) / np.pi

    return np.stack([u, v], axis=1).astype(np.float64)


def write_obj_with_texture(mesh: trimesh.Trimesh,
                          texture_image: np.ndarray,
                          output_path: str) -> bool:
    try:
        base_path = output_path.replace('.obj', '').replace('.mtl', '')
        obj_path = f"{base_path}.obj"
        mtl_path = f"{base_path}.mtl"
        tex_path = f"{base_path}.png"

        mesh.export(obj_path)
        logger.info(f"Exported OBJ to {obj_path}")

        mtl_content = f"""# Material file for {os.path.basename(obj_path)}
newmtl garment_material
Ka 1.0 1.0 1.0
Kd 0.8 0.8 0.8
Ks 0.2 0.2 0.2
Ns 32.0
map_Kd {os.path.basename(tex_path)}
"""
        with open(mtl_path, 'w') as f:
            f.write(mtl_content)

        with open(obj_path, 'r') as f:
            obj_content = f.read()

        mtl_line = f"mtllib {os.path.basename(mtl_path)}\n"
        usemtl_line = "usemtl garment_material\n"

        if "mtllib" not in obj_content:
            obj_content = mtl_line + obj_content
        if "usemtl" not in obj_content:
            obj_content = obj_content.replace("f ", usemtl_line + "f ", 1)

        with open(obj_path, 'w') as f:
            f.write(obj_content)

        if texture_image is not None:
            if texture_image.dtype == np.float32:
                texture_image = (texture_image * 255).astype(np.uint8)
            img = Image.fromarray(texture_image, mode='RGB')
            img.save(tex_path)

        return True

    except Exception as e:
        logger.error(f"Failed to export OBJ with texture: {e}")
        raise


def write_glb_with_texture(mesh: trimesh.Trimesh,
                          texture_image: np.ndarray,
                          output_path: str) -> bool:
    try:
        if not output_path.endswith('.glb'):
            output_path = output_path.replace('.obj', '').replace('.mtl', '') + '.glb'

        if texture_image is not None:
            if texture_image.dtype in (np.float32, np.float64):
                if texture_image.max() <= 1.0:
                    texture_image = (texture_image * 255).astype(np.uint8)
                else:
                    texture_image = texture_image.astype(np.uint8)

            if texture_image.shape[-1] == 3:
                pil_texture = Image.fromarray(texture_image, mode='RGB').convert('RGBA')
            elif texture_image.shape[-1] == 4:
                pil_texture = Image.fromarray(texture_image, mode='RGBA')
            else:
                pil_texture = Image.fromarray(texture_image).convert('RGBA')

            verts = mesh.vertices
            uv_coords = _generate_spherical_uv(verts)
            uv_coords = np.clip(uv_coords, 0.0, 1.0)

            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=pil_texture,
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.8,
                doubleSided=True,
            )

            texture_visuals = trimesh.visual.TextureVisuals(
                uv=uv_coords,
                material=material,
            )
            mesh.visual = texture_visuals

        glb_data = mesh.export(file_type='glb')

        with open(output_path, 'wb') as f:
            f.write(glb_data)

        file_size = len(glb_data)
        logger.info(f"Exported GLB to {output_path} ({file_size} bytes)")

        if file_size < 5000:
            logger.warning(f"GLB file is very small ({file_size} bytes)")

        return True

    except Exception as e:
        logger.error(f"Failed to export GLB: {e}")
        raise


def export_3d_model(body_mesh: Meshes,
                    garment_mesh: Meshes,
                    output_path: str,
                    vertex_colors: Optional[np.ndarray] = None,
                    texture_image: Optional[np.ndarray] = None,
                    texture_quality: str = "fast") -> bool:
    try:
        if body_mesh is None or garment_mesh is None:
            raise ValueError("Body mesh or garment mesh is None")

        if body_mesh.isempty() or garment_mesh.isempty():
            raise ValueError("Body mesh or garment mesh is empty")

        body_verts = body_mesh.verts_list()
        garment_verts = garment_mesh.verts_list()

        if not body_verts or not garment_verts:
            raise ValueError("Body or garment mesh has no vertices")

        logger.info(
            f"Combining meshes: body ({len(body_verts[0])} verts) + "
            f"garment ({len(garment_verts[0])} verts)"
        )

        combined_mesh = join_meshes_as_scene(
            [body_mesh.cpu(), garment_mesh.cpu()]
        )

        verts_list = combined_mesh.verts_list()
        faces_list = combined_mesh.faces_list()

        verts = verts_list[0].cpu().numpy()
        faces = faces_list[0].cpu().numpy()

        logger.info(f"Combined mesh: {len(verts)} vertices, {len(faces)} faces")

        mesh_scene = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        if texture_image is None and vertex_colors is not None:
            if vertex_colors.dtype in (np.float32, np.float64):
                if vertex_colors.max() <= 1.0:
                    colors_uint8 = (vertex_colors * 255).astype(np.uint8)
                else:
                    colors_uint8 = vertex_colors.astype(np.uint8)
            else:
                colors_uint8 = vertex_colors.astype(np.uint8)

            if len(colors_uint8) == len(verts):
                mesh_scene.visual.vertex_colors = colors_uint8

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        glb_path = output_path.rstrip('/') + '.glb'
        write_glb_with_texture(mesh_scene, texture_image, glb_path)

        if texture_image is not None:
            obj_path = output_path.rstrip('/') + '.obj'
            write_obj_with_texture(mesh_scene, texture_image, obj_path)

        if not os.path.exists(glb_path):
            raise IOError(f"File was not created at {glb_path}")

        logger.info(f"Successfully exported 3D model to {glb_path}")
        return True

    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise
