# Copyright (C) 2025 Malcom3D <malcom3d.gpl@gmail.com>
#
# This file is part of pbrAudio.
#
# pbrAudio is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pbrAudio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pbrAudio.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

from core.entity_manager import EntityManager
from lib.functions import _audio_to_npz, _get_position, _world_to_grid, _is_in_bounds, _cartesian_to_spherical

@dataclass
class AcousticObject:
    """
    Acoustic object
    """
    entity_manager: EntityManager
    idx: int

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.frames = self.entity_manager.get('frames')
        self.shape = config.acoustic_domain.shape
        self.voxel_size = config.acoustic_domain.voxel_size
        self.grid_geometry = config.acoustic_domain.geometry

        for object_config in config.objects:
            if object_config.idx == self.idx:
                self.object_config = object_config

    def get_soxels(self):
        """Voxelize an acoustic object into the grid"""
        # Load object mesh from OBJ files
        mesh_vertices, mesh_faces = self._load_object_mesh()

        if mesh_vertices is None or mesh_faces is None:
            print(f"Warning: Could not load mesh for object: {self.object_config.name}")
            return

        # Voxelize mesh
        object_voxels = self._voxelize_mesh(mesh_vertices, mesh_faces)

        # Update soxels at object positions
        soxels = []
        for i, j, k in object_voxels:
            if _is_in_bounds(self.shape, i, j, k):
                soxel = Soxel(
                    idx=self.object_config.idx,
                    type = 2,                  # mark as object
                    input_pressures = None,
                    acoustic_shader = self.object_config.acoustic_shader
                )
                soxels.append([[i,j,k],soxel])
        return soxels

    def _load_object_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load object mesh from OBJ files"""
        # This is a simplified implementation
        # In practice, you'd need to handle proper OBJ parsing

        current_frame = self.frames.get()

        if not self.object_config.obj_files:
            return None, None
    
        try:
            # Simple OBJ parser for demonstration
            vertices = []
            faces = []
    
            obj_file = self.object_config.obj_files[current_frame]
            with open(obj_file, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        # Vertex
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                            vertex = _world_to_grid(self.voxel_size, self.grid_geometry, vertex)
                            vertices.append(vertex)
                    elif line.startswith('f '):
                        # Face
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            # Handle different face formats (v, v/vt, v/vt/vn)
                            face_vertices = []
                            for part in parts[1:]:
                                vertex_index = part.split('/')[0]
                                if vertex_index:
                                    face_vertices.append(int(vertex_index) - 1)  # OBJ is 1-indexed
                            if len(face_vertices) >= 3:
                                faces.append(face_vertices)

            return np.array(vertices, dtype=np.int32), np.array(faces, dtype=np.int32)

        except Exception as e:
            print(f"Warning: Failed to load OBJ file {self.object_config.obj_files[0]}: {e}")
            return None, None

    def _voxelize_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> List[Tuple[int, int, int]]:
        """Voxelize a mesh into grid positions"""
        # Simple voxelization using bounding box and point-in-mesh test
        # In practice, use more sophisticated voxelization algorithms
        voxels = []

        if vertices is None or len(vertices) == 0:
            return voxels

        # Calculate bounding box
        min_grid = np.min(vertices, axis=0)
        max_grid = np.max(vertices, axis=0)

        # Simple voxelization: check if voxel center is inside mesh
        for i in range(max(0, min_grid[0]), min(self.shape[0], max_grid[0] + 1)):
            for j in range(max(0, min_grid[1]), min(self.shape[1], max_grid[1] + 1)):
                for k in range(max(0, min_grid[2]), min(self.shape[2], max_grid[2] + 1)):
                    voxel_center = (i, j, k)

                    if self._is_point_in_mesh(voxel_center, vertices, faces):
                        if _is_in_bounds(self.shape, i, j, k):
                            voxels.append((i, j, k))
        return voxels

    def _is_point_in_mesh(self, point: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> bool:
        """Check if a point is inside a mesh using ray casting"""
        # Simplified point-in-mesh test
        # In practice, use more robust algorithms
        ray_direction = np.array([1.0, 0.0, 0.0])  # Arbitrary direction
        intersection_count = 0

        for face in faces:
            if len(face) >= 3:
                triangle = vertices[face[:3]]
                if self._ray_triangle_intersection(point, ray_direction, triangle):
                    intersection_count += 1

        return intersection_count % 2 == 1  # Odd number of intersections = inside

    def _ray_triangle_intersection(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                                 triangle: np.ndarray) -> bool:
        """Check if ray intersects triangle (Möller–Trumbore algorithm)"""
        # Simplified implementation
        # In practice, use the full Möller–Trumbore algorithm
        epsilon = 1e-6

        edge1 = triangle[1] - triangle[0]
        edge2 = triangle[2] - triangle[0]
        h = np.cross(ray_dir, edge2)
        a = np.dot(edge1, h)

        if abs(a) < epsilon:
            return False  # Ray parallel to triangle

        f = 1.0 / a
        s = ray_origin - triangle[0]
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return False

        q = np.cross(s, edge1)
        v = f * np.dot(ray_dir, q)

        if v < 0.0 or u + v > 1.0:
            return False

        t = f * np.dot(edge2, q)

        return t > epsilon
