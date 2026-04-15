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
import numba as nb
import numpy as np
import trimesh
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..core.entity_manager import EntityManager
from ..lib.ray_data import RayData
from ..lib.functions import _load_mesh

class RayTracer:
    """ Ray tracing engine using trimesh with embree support """
    entity_manager: EntityManager
    
    def __init__(self, entity_manager: EntityManager):
        self.entity_manager = entity_manager
        self.config = entity_manager.get('config')
        
    def intersect_ray(self, origin: np.ndarray, direction: np.ndarray, scene_meshes: List[trimesh.Trimesh], scene_meshes_ids: List[int], max_distance: float = np.inf) -> Dict:
        """
        Find the closest intersection of a ray with all objects.
            - scene_meshes: list of trimesh mesh in the scene
            - scene_meshes_ids: list of idx of the mesh in the scene
        Returns a dict with:
            - 'hit': bool
            - 'object_idx': index of the object hit
            - 'point': intersection point
            - 'normal': normal at intersection
            - 'distance': distance from origin
        """
        # Ensure direction is normalized
        direction = direction / np.linalg.norm(direction)
        
        # Use trimesh's ray-mesh intersection
        # We need to iterate over scene_meshes to find the closest hit
        closest_dist = np.inf
        hit_info = None
        for mesh, obj_idx in zip(scene_meshes, scene_meshes_ids):
            # Use ray.intersects_location
            locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=[origin],
                ray_directions=[direction],
                multiple_hits=False
            )
            if len(locations) > 0:
                # Compute distance
                dist = np.linalg.norm(locations[0] - origin)
                if dist < closest_dist and dist <= max_distance:
                    closest_dist = dist
                    # Get normal at intersection
                    face_normal = mesh.face_normals[index_tri[0]]
                    hit_info = {
                        'hit': True,
                        'object_idx': obj_idx,
                        'point': locations[0],
                        'normal': face_normal,
                        'distance': dist
                    }
        if hit_info:
            return hit_info
        else:
            return {'hit': False, 'object_idx': obj_idx}
