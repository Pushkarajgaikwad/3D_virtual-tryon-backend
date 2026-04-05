"""
TemplateManager — loads and manages garment template meshes.

Templates live under  <templates_root>/<template_id>/
Each template directory may contain:
    mesh.obj | mesh.npz   — geometry
    uv.npy               — UV coordinates  (optional)

If no templates are found on disk a single built-in synthetic template
("shirt_default") is registered so the server can still start.
"""

import logging
import os
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
import torch
import trimesh

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — synthetic garment mesh
# ---------------------------------------------------------------------------

def _make_cylinder_mesh(
    radius: float = 0.2,
    height: float = 0.6,
    sections: int = 16,
) -> trimesh.Trimesh:
    """Return a simple open cylinder trimesh (shirt-like silhouette)."""
    return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)


def _mesh_to_edge_index(faces: np.ndarray) -> torch.Tensor:
    """Convert (F, 3) face array to (2, E) COO edge index (undirected)."""
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    src, dst = zip(*edges)
    # Undirected: add both directions
    src = list(src) + list(dst)
    dst = list(dst) + list(src[:len(dst)])
    return torch.tensor([src, dst], dtype=torch.long)


# ---------------------------------------------------------------------------
# TemplateManager
# ---------------------------------------------------------------------------

class TemplateManager:
    """
    Manages garment template meshes used for draping.

    Args:
        templates_root: Directory that contains one sub-folder per template.
    """

    def __init__(self, templates_root: str = "templates"):
        self.templates_root = templates_root
        self._templates: Dict[str, dict] = {}  # template_id → data dict
        self._embeddings: Dict[str, torch.Tensor] = {}

        self._discover_templates()

        if not self._templates:
            logger.warning(
                f"No templates found in '{templates_root}'. "
                "Registering built-in synthetic template."
            )
            self._register_synthetic()

        logger.info(
            f"TemplateManager ready: {list(self._templates.keys())}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_templates(self) -> List[str]:
        return list(self._templates.keys())

    def load_template(self, template_id: str) -> dict:
        """
        Return template data dict with keys:
            vertices  – (N, 3) float32 tensor
            edges     – (2, E) long tensor  (COO edge index)
            faces     – (F, 3) long tensor | None
        """
        if template_id not in self._templates:
            available = list(self._templates.keys())
            raise KeyError(
                f"Template '{template_id}' not found. Available: {available}"
            )
        return self._templates[template_id]

    def get_template_embedding(
        self, template_id: str, emb_dim: int = 16
    ) -> torch.Tensor:
        """
        Return a fixed (emb_dim,) embedding for the template.
        Derived deterministically from the template ID string hash.
        """
        key = (template_id, emb_dim)
        if key not in self._embeddings:
            rng = np.random.RandomState(abs(hash(template_id)) % (2**31))
            emb = rng.randn(emb_dim).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            self._embeddings[key] = torch.tensor(emb)
        return self._embeddings[key]

    def select_template_by_image(self, garment_bytes: bytes) -> str:
        """
        Heuristic: pick a template based on garment image aspect ratio.
        Tall images → first template; wide → last; square → middle.
        """
        from PIL import Image
        img = Image.open(BytesIO(garment_bytes))
        w, h = img.size
        templates = self.list_templates()
        if not templates:
            raise ValueError("No templates available")
        ratio = h / max(w, 1)
        if ratio > 1.2:
            return templates[0]
        elif ratio < 0.8:
            return templates[-1]
        else:
            return templates[len(templates) // 2]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _discover_templates(self):
        if not os.path.isdir(self.templates_root):
            return

        for name in sorted(os.listdir(self.templates_root)):
            folder = os.path.join(self.templates_root, name)
            if not os.path.isdir(folder):
                continue
            try:
                data = self._load_from_folder(folder)
                self._templates[name] = data
                logger.info(f"Loaded template '{name}' ({data['vertices'].shape[0]} verts)")
            except Exception as e:
                logger.warning(f"Skipping template '{name}': {e}")

    def _load_from_folder(self, folder: str) -> dict:
        # Try .npz first, then .obj
        npz_path = os.path.join(folder, "mesh.npz")
        obj_path = os.path.join(folder, "mesh.obj")

        if os.path.exists(npz_path):
            data = np.load(npz_path)
            verts = torch.tensor(data["vertices"], dtype=torch.float32)
            faces_np = data["faces"].astype(np.int64) if "faces" in data else None
        elif os.path.exists(obj_path):
            mesh = trimesh.load(obj_path, force="mesh", process=False)
            verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
            faces_np = np.array(mesh.faces, dtype=np.int64)
        else:
            raise FileNotFoundError("No mesh.npz or mesh.obj found")

        faces = torch.tensor(faces_np, dtype=torch.long) if faces_np is not None else None
        edges = _mesh_to_edge_index(faces_np if faces_np is not None else np.empty((0, 3), dtype=np.int64))
        return {"vertices": verts, "edges": edges, "faces": faces}

    def _register_synthetic(self):
        """Register a built-in cylinder mesh as 'shirt_default'."""
        mesh = _make_cylinder_mesh()
        verts = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
        faces_np = np.array(mesh.faces, dtype=np.int64)
        faces = torch.tensor(faces_np, dtype=torch.long)
        edges = _mesh_to_edge_index(faces_np)
        self._templates["shirt_default"] = {
            "vertices": verts,
            "edges": edges,
            "faces": faces,
        }
