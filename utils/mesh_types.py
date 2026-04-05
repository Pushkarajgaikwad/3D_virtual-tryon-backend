"""
Lightweight Meshes shim — drop-in replacement for pytorch3d.structures.Meshes
for the subset of the API used in this project.

Avoids the pytorch3d build dependency while keeping the same call sites.
"""

from __future__ import annotations

from typing import List, Optional

import torch


class Meshes:
    """
    Minimal replacement for pytorch3d.structures.Meshes.

    Supports:
        Meshes(verts=[...], faces=[...])
        .verts_list()  → List[Tensor]
        .faces_list()  → List[Tensor]
        .isempty()     → bool
        .cpu()         → Meshes (all tensors moved to CPU)
    """

    def __init__(
        self,
        verts: List[torch.Tensor],
        faces: List[torch.Tensor],
    ):
        self._verts = verts
        self._faces = faces

    # ------------------------------------------------------------------

    def verts_list(self) -> List[torch.Tensor]:
        return self._verts

    def faces_list(self) -> List[torch.Tensor]:
        return self._faces

    def isempty(self) -> bool:
        if not self._verts:
            return True
        return all(v.numel() == 0 for v in self._verts)

    def cpu(self) -> "Meshes":
        return Meshes(
            verts=[v.cpu() for v in self._verts],
            faces=[f.cpu() for f in self._faces],
        )

    def __repr__(self) -> str:
        n = len(self._verts)
        sizes = [v.shape[0] for v in self._verts] if self._verts else []
        return f"Meshes(batch_size={n}, num_verts={sizes})"


def join_meshes_as_scene(meshes: List[Meshes]) -> Meshes:
    """
    Concatenate a list of Meshes objects into a single-mesh scene.

    Vertex indices in each mesh's faces are offset by the cumulative
    vertex count of preceding meshes.
    """
    all_verts: List[torch.Tensor] = []
    all_faces: List[torch.Tensor] = []

    offset = 0
    for mesh in meshes:
        for v, f in zip(mesh.verts_list(), mesh.faces_list()):
            all_verts.append(v)
            if f.numel() > 0:
                all_faces.append(f + offset)
            offset += v.shape[0]

    if not all_verts:
        return Meshes(verts=[], faces=[])

    combined_verts = torch.cat(all_verts, dim=0)
    combined_faces = torch.cat(all_faces, dim=0) if all_faces else torch.zeros((0, 3), dtype=torch.long)

    return Meshes(verts=[combined_verts], faces=[combined_faces])
