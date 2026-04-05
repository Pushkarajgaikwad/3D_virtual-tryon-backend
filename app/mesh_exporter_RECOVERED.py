import torch
import trimesh
import numpy as np
import os
from pytorch3d.structures import Meshes
from pytorch3d.ops import join_meshes_as_scene
from typing import List, Optional
import logging

# Setup logging
logger = logging.getLogger(__name__)

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
            from PIL import Image
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
    Export mesh to GLB format with embedded texture and vertex colors.
    
    GLB is a binary GLTF format that supports:
    - Vertex colors
    - Embedded textures
    - Advanced materials
    
    Args:
        mesh: trimesh.Trimesh object.
        texture_image: Texture image (H, W, 3) uint8 or None.
        output_path: Path to save .glb file.
    
    Returns:
        True if successful.
    """
    try:
        # Ensure extension is .glb
        if not output_path.endswith('.glb'):
            output_path = output_path.replace('.obj', '').replace('.mtl', '') + '.glb'
        
        # Export with trimesh (handles vertex colors automatically)
        mesh.export(output_path, file_type='glb')
        
        file_size = os.path.getsize(output_path)
        logger.info(f"Exported GLB to {output_path} ({file_size} bytes)")
        
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
    - .glb format with vertex colors and embedded textures
    - .obj + .mtl + .png format (advanced material support)
    - Vertex color mapping for per-vertex coloring
    
    Workflow:
    1. Join body and garment meshes into single scene
    2. Apply vertex colors if provided
    3. Export to specified format (GLB or OBJ)
    
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
        try:
            mesh_scene = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        except Exception as e:
            raise RuntimeError(f"Error creating Trimesh object: {e}")
        
        # Apply vertex colors if provided
        if vertex_colors is not None:
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