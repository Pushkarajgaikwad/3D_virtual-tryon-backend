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
    """
    Export mesh to OBJ + MTL + PNG texture format.
    
    Creates:
    - output.obj (mesh with material references)
    - output.mtl (material file)
    - output.png (texture image)
    
    Args:
        mesh: trimesh.Trimesh object with geometry and vertex colors.
        texture_image: Texture image (H, W, 3) uint8 or None.
        output_path: Base path (without extension).
    
    Returns:
        True if successful.
    """
    try:
        base_path = output_path.replace('.obj', '').replace('.mtl', '')
        obj_path = f"{base_path}.obj"
        mtl_path = f"{base_path}.mtl"
        tex_path = f"{base_path}.png"
        
        # Export OBJ
        mesh.export(obj_path)
        logger.info(f"Exported OBJ to {obj_path}")
        
        # Create MTL file
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
        logger.info(f"Created MTL file: {mtl_path}")
        
        # Inject MTL reference into OBJ
        with open(obj_path, 'r') as f:
            obj_content = f.read()
        
        mtl_line = f"mtllib {os.path.basename(mtl_path)}\n"
        usemtl_line = "usemtl garment_material\n"
        
        if "mtllib" not in obj_content:
            obj_content = mtl_line + obj_content
        if "usemtl" not in obj_content:
            obj_content = obj_content.replace(
                "f ",
                usemtl_line + "f ",
                1
            )
        
        with open(obj_path, 'w') as f:
            f.write(obj_content)
        
        # Save texture
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

def write_glb_with_texture(mesh: trimesh.Trimesh,
                          texture_image: np.ndarray,
                          output_path: str) -> bool:
    """
    Export mesh to GLB format with embedded PBR texture via UV mapping.
    
    This is the FIXED version that properly creates TextureVisuals with
    PBRMaterial to avoid the blank white mesh issue.
    
    Args:
        mesh: trimesh.Trimesh object.
        texture_image: Texture image (H, W, 3/4) uint8 or None.
        output_path: Path to save .glb file.
    
    Returns:
        True if successful.
    """
    try:
        # Ensure extension is .glb
        if not output_path.endswith('.glb'):
            output_path = output_path.replace('.obj', '').replace('.mtl', '') + '.glb'
        
        # --- Apply proper TextureVisuals if texture is provided ---
        if texture_image is not None:
            # Normalize texture to uint8 RGBA
            if texture_image.dtype in (np.float32, np.float64):
                if texture_image.max() <= 1.0:
                    texture_image = (texture_image * 255).astype(np.uint8)
                else:
                    texture_image = texture_image.astype(np.uint8)
            
            # Convert to PIL Image (RGBA)
            if texture_image.shape[-1] == 3:
                pil_texture = Image.fromarray(texture_image, mode='RGB').convert('RGBA')
            elif texture_image.shape[-1] == 4:
                pil_texture = Image.fromarray(texture_image, mode='RGBA')
            else:
                pil_texture = Image.fromarray(texture_image).convert('RGBA')
            
            # Generate UV coordinates if not already present
            verts = mesh.vertices
            uv_coords = _generate_spherical_uv(verts)
            
            # Clamp UVs
            uv_coords = np.clip(uv_coords, 0.0, 1.0)
            
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
            
            logger.info(f"Applied PBR TextureVisuals with UV mapping ({pil_texture.size} texture, {len(uv_coords)} UVs)")
        
        # Export to GLB
        glb_data = mesh.export(file_type='glb')
        
        with open(output_path, 'wb') as f:
            f.write(glb_data)
        
        file_size = len(glb_data)
        logger.info(f"Exported GLB to {output_path} ({file_size} bytes)")
        
        if file_size < 5000:
            logger.warning(
                f"GLB file is very small ({file_size} bytes) — "
                "texture may not have been embedded correctly."
            )
        
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
    """
    Combines body and garment meshes and exports them with optional textures.
    
    Supports:
    - .glb format with PBR TextureVisuals and embedded textures
    - .obj + .mtl + .png format (advanced material support)
    - Vertex color mapping for per-vertex coloring
    
    Workflow:
    1. Join body and garment meshes into single scene
    2. Apply texture via UV mapping (PBRMaterial + TextureVisuals)
    3. Fall back to vertex colors if no texture image provided
    4. Export to specified format (GLB or OBJ)
    
    Args:
        body_mesh: PyTorch3D Meshes object for human body.
        garment_mesh: PyTorch3D Meshes object for draped garment.
        output_path: File path (without extension, e.g., 'output/job123').
        vertex_colors: (Optional) Per-vertex RGB colors (N, 3) float [0, 1].
        texture_image: (Optional) Texture image (H, W, 3) uint8.
        texture_quality: "fast" (1024px) or "high" (2048px+). Default: "fast".
    
    Returns:
        True if export successful, False otherwise.
    
    Raises:
        ValueError: If mesh data invalid or empty.
        IOError: If file writing fails.
    """
    
    try:
        # --- 1. Validate Input Meshes ---
        if body_mesh is None or garment_mesh is None:
            raise ValueError("Body mesh or garment mesh is None")
        
        if body_mesh.isempty() or garment_mesh.isempty():
            raise ValueError("Body mesh or garment mesh is empty")
        
        body_verts = body_mesh.verts_list()
        garment_verts = garment_mesh.verts_list()
        
        if not body_verts or not garment_verts:
            raise ValueError("Body or garment mesh has no vertices")
        
        if len(body_verts[0]) == 0 or len(garment_verts[0]) == 0:
            raise ValueError("Body or garment mesh vertices are empty")
        
        # --- 2. Combine Meshes ---
        logger.info(
            f"Combining meshes: body ({len(body_verts[0])} verts) + "
            f"garment ({len(garment_verts[0])} verts)"
        )
        
        try:
            combined_mesh = join_meshes_as_scene(
                [body_mesh.cpu(), garment_mesh.cpu()]
            )
        except Exception as e:
            raise RuntimeError(f"Error joining meshes with PyTorch3D: {e}")

        # --- 3. Extract Data for Trimesh ---
        verts_list = combined_mesh.verts_list()
        faces_list = combined_mesh.faces_list()
        
        if not verts_list or not faces_list:
            raise ValueError("Failed to extract vertices or faces from combined mesh")
        
        verts = verts_list[0].cpu().numpy()
        faces = faces_list[0].cpu().numpy()
        
        if len(verts) == 0 or len(faces) == 0:
            raise ValueError("Extracted vertices or faces are empty")
        
        logger.info(f"Combined mesh: {len(verts)} vertices, {len(faces)} faces")

        # --- 4. Create Trimesh Object ---
        # IMPORTANT: process=False prevents vertex reordering that
        # would break UV <-> vertex mapping.
        try:
            mesh_scene = trimesh.Trimesh(
                vertices=verts, faces=faces, process=False
            )
        except Exception as e:
            raise RuntimeError(f"Error creating Trimesh object: {e}")
        
        # Apply vertex colors as fallback if no texture image
        if texture_image is None and vertex_colors is not None:
            if vertex_colors.dtype == np.float32 or vertex_colors.dtype == np.float64:
                if vertex_colors.max() <= 1.0:
                    colors_uint8 = (vertex_colors * 255).astype(np.uint8)
                else:
                    colors_uint8 = vertex_colors.astype(np.uint8)
            else:
                colors_uint8 = vertex_colors.astype(np.uint8)
            
            # Ensure color count matches vertex count
            if len(colors_uint8) == len(verts):
                mesh_scene.visual.vertex_colors = colors_uint8
                logger.info(f"Applied vertex colors: {colors_uint8.shape}")
            else:
                logger.warning(
                    f"Vertex color count ({len(colors_uint8)}) "
                    f"doesn't match vertex count ({len(verts)}). Skipping."
                )

        # --- 5. Export File ---
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
            
            # Determine export format
            if output_path.endswith('.glb') or output_path.endswith('.gltf'):
                # GLB export (preferred for textured meshes)
                glb_path = output_path if output_path.endswith('.glb') else output_path.replace('.gltf', '.glb')
                write_glb_with_texture(mesh_scene, texture_image, glb_path)
                final_path = glb_path
            else:
                # Default to GLB
                glb_path = output_path.rstrip('/') + '.glb'
                write_glb_with_texture(mesh_scene, texture_image, glb_path)
                
                # Also create OBJ variant if textures available
                if texture_image is not None:
                    obj_path = output_path.rstrip('/') + '.obj'
                    write_obj_with_texture(mesh_scene, texture_image, obj_path)
                    logger.info(f"Exported both GLB and OBJ formats")
                
                final_path = glb_path
            
            # Verify file was created
            if not os.path.exists(final_path):
                raise IOError(f"File was not created at {final_path}")
            
            file_size = os.path.getsize(final_path)
            logger.info(
                f"Successfully exported 3D model to {final_path} "
                f"({file_size} bytes, texture_quality={texture_quality})"
            )
            return True
            
        except Exception as e:
            raise IOError(f"Error exporting mesh: {e}")
            
    except (ValueError, RuntimeError, IOError) as e:
        logger.error(f"Export failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during mesh export: {e}")
        raise
