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
import trimesh
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

from core.entity_manager import EntityManager
from lib.soxel import Soxel
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
        # Load object mesh from OBJ files using trimesh
        mesh = self._load_object_mesh()

        if mesh is None:
            print(f"Warning: Could not load mesh for object: {self.object_config.name}")
            return

        # Voxelize mesh using trimesh
        object_voxels = self._voxelize_mesh(mesh)

        # Update soxels at object positions
        soxels = []
        for i, j, k in object_voxels:
            if _is_in_bounds(self.shape, i, j, k):
                soxel = Soxel(
                    idx=self.object_config.idx,
                    type=2,  # mark as object
                    input_pressures=None,
                    acoustic_shader=self.object_config.acoustic_shader
                )
                soxels.append([i, j, k, soxel])
        return soxels

    def _load_object_mesh(self) -> Optional[trimesh.Trimesh]:
        """Load object mesh from OBJ files using trimesh"""
        current_frame = self.frames.get()

        if not self.object_config.obj_files:
            return None

        try:
            obj_files = self.object_config.obj_files
            if not len(obj_files) == 1:
                obj_file = obj_files[current_frame]
            else:
                obj_file = obj_files[0]
            
            # Load mesh using trimesh
            mesh = trimesh.load_mesh(obj_file)
            
            # Ensure we have a triangular mesh
            if not isinstance(mesh, trimesh.Trimesh):
                # If it's a scene or other type, try to extract the first mesh
                if hasattr(mesh, 'geometry'):
                    mesh = list(mesh.geometry.values())[0]
                else:
                    print(f"Warning: Unsupported mesh type for {obj_file}")
                    return None
            
            # Convert vertices to grid coordinates
            vertices_world = mesh.vertices
            vertices_grid = np.array([
                _world_to_grid(self.voxel_size, self.grid_geometry, vertex) 
                for vertex in vertices_world
            ], dtype=np.int32)
            
            # Create new mesh with grid coordinates
            mesh_grid = trimesh.Trimesh(vertices=vertices_grid, faces=mesh.faces)
            
            return mesh_grid

        except Exception as e:
            print(f"Warning: Failed to load OBJ file {self.object_config.obj_files[current_frame]}: {e}")
            return None

    def _voxelize_mesh(self, mesh: trimesh.Trimesh) -> List[Tuple[int, int, int]]:
        """Voxelize a mesh into grid positions using trimesh voxelization"""
        voxels = []

        if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
            return voxels

        try:
            # Method 1: Use trimesh's built-in voxelization
            # This creates a voxel grid that fits the mesh
            voxel_grid = mesh.voxelized(pitch=1.0)  # pitch=1.0 since we're in grid coordinates
            
            # Get the voxel centers in grid coordinates
            if voxel_grid is not None and voxel_grid.shape is not None:
                # voxel_grid.matrix is a 3D boolean array where True indicates occupied voxels
                occupied_voxels = np.argwhere(voxel_grid.matrix)
                
                # Convert to global grid coordinates
                # The voxel grid might have its own origin, so we need to transform
#                voxel_origin = voxel_grid.origin
                for voxel in occupied_voxels:
                    global_coords = (
                        int(voxel[0]),
                        int(voxel[1]), 
                        int(voxel[2])
                    )
#                    global_coords = (
#                        int(voxel[0] + voxel_origin[0]),
#                        int(voxel[1] + voxel_origin[1]), 
#                        int(voxel[2] + voxel_origin[2])
#                    )
                    voxels.append(global_coords)
                    
        except Exception as e:
            print(f"Warning: Trimesh voxelization failed, falling back to bounding box method:: {e}")
            # Fallback to bounding box method
            voxels = self._voxelize_mesh_bounding_box(mesh)

        return voxels

    def _voxelize_mesh_bounding_box(self, mesh: trimesh.Trimesh) -> List[Tuple[int, int, int]]:
        """Fallback voxelization using bounding box and point-in-mesh test"""
        voxels = []
        
        # Get bounding box in grid coordinates
        bbox = mesh.bounds
        min_grid = np.floor(bbox[0]).astype(int)
        max_grid = np.ceil(bbox[1]).astype(int)
        
        # Clamp to grid boundaries
        min_grid = np.maximum(min_grid, [0, 0, 0])
        max_grid = np.minimum(max_grid, self.shape)
        
        # Check each voxel in the bounding box
        for i in range(min_grid[0], max_grid[0] + 1):
            for j in range(min_grid[1], max_grid[1] + 1):
                for k in range(min_grid[2], max_grid[2] + 1):
                    point = np.array([i, j, k])
                    
                    # Use trimesh's contains_points method for robust point-in-mesh test
                    if mesh.contains([point])[0]:
                        voxels.append((i, j, k))
        
        return voxels

    def _voxelize_mesh_raycasting(self, mesh: trimesh.Trimesh) -> List[Tuple[int, int, int]]:
        """Alternative voxelization using ray casting (more accurate but slower)"""
        voxels = []
        
        # Get bounding box
        bbox = mesh.bounds
        min_grid = np.floor(bbox[0]).astastype(int)
        max_grid = np.ceil(bbox[1]).astype(int)
        
        # Clamp to grid boundaries
        min_grid = np.maximum(min_grid, [0, 0, 0])
        max_grid = np.minimum(max_grid, self.shape)
        
        # Create ray origins for each x-z plane
        ray_origins = []
        grid_points = []
        
        for i in range(min_grid[0], max_grid[0] + 1):
            for k in range(min_grid[2], max_grid[2] + 1):
                # Create ray from bottom to top of bounding box
                origin = np.array([i, min_grid[1] - 1, k])
                ray_origins.append(origin)
                grid_points.append((i, k))
        
        if ray_origins:
            ray_origins = np.array(ray_origins)
            ray_directions = np.array([[0, 1, 0]] * len(ray_origins))  # All rays point upward
            
            # Find intersections with mesh
            locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins, ray_directions, multiple_hits=True
            )
            
            # Process intersections to determine filled voxels
            if len(locations) > 0:
                voxels = self._process_ray_intersections(
                    locations, index_ray, grid_points, min_grid, max_grid
                )
        
        return voxels

    def _process_ray_intersections(self, locations: np.ndarray, index_ray: np.ndarray, 
                                 grid_points: List[Tuple[int, int]], 
                                 min_grid: np.ndarray, max_grid: np.ndarray) -> List[Tuple[int, int, int]]:
        """Process ray intersections to determine filled voxels"""
        voxels = []
        
        # Group intersections by ray
        ray_intersections = {}
        for loc, ray_idx in zip(locations, index_ray):
            if ray_idx not in ray_intersections:
                ray_intersections[ray_idx] = []
            ray_intersections[ray_idx].append(loc[1])  # y-coordinate
        
        # For each ray, sort intersections and apply even-odd rule
        for ray_idx, y_coords in ray_intersections.items():
            y_coords.sort()
            
            # Even-odd rule: voxels between pairs of intersections are filled
            for pair_idx in range(0, len(y_coords) - 1, 2):
                y_start = max(min_grid[1], int(np.floor(y_coords[pair_idx])))
                y_end = min(max_grid[1], int(np.ceil(y_coords[pair_idx + 1])))
                
                i, k = grid_points[ray_idx]
                for j in range(y_start, y_end + 1):
                    voxels.append((i, j, k))
        
        return voxels
