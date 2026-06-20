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
from pbrAudioCommon.lib.import_helper import np
import trimesh
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.soxel import Soxel
from ..lib.functions import _audio_to_npz, _get_position, _world_to_grid, _is_in_bounds, _cartesian_to_spherical

# To be reworked using trimesh voxelized

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
            
            return mesh

        except Exception as e:
            print(f"Warning: Failed to load OBJ file {self.object_config.obj_files[current_frame]}: {e}")
            return None

    def _voxelize_mesh(self, mesh: trimesh.Trimesh) -> List[Tuple[int, int, int]]:
        """Voxelize a mesh into grid positions using trimesh"""

        if mesh is None or mesh.vertices is None:
            return voxels

        vertices = mesh.vertices
        voxels = _world_to_grid(self.voxel_size, self.grid_geometry, vertices)
                    
        return voxels.tolist()
