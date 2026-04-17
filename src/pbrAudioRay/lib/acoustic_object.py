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
from typing import Tuple, Optional, List, Any
from dataclasses import dataclass, field

from rigidBody import Pym2f

from ..core.entity_manager import EntityManager
from ..lib.functions import _load_mesh

@dataclass
class AcousticObject:
    """ Handle mesh object geometry and modal model analysis """
    entity_manager: EntityManager
    config_obj: Any
    obj_idx: int = None

    def __post_init__(self):
        self.obj_idx = self.config_obj.idx

    def get_mesh(self, frame_idx: int, source_pos: np.ndarray, output_pos: np.ndarray) -> trimesh.Trimesh:
        """ Return mesh object geometry refined using trimesh simplify_quadric_decimation if distance from sources and listeners is greater than a threshold """
        vertices, normals, faces = _load_mesh(self.config_obj, frame_idx)
        mesh = trimesh.Trimesh(vertices=vertices, vertex_normals=normals, faces=faces)

        config = self.entity_manager.get('config')
        adr_threshold = config.system.adr_threshold

        # If no ADR threshold, return full mesh
        if adr_threshold is None:
            return mesh

        # Calculate distances from object to source and output
        # Use object's bounding sphere center as reference point
        obj_center = mesh.bounding_sphere.center

        # Calculate minimum distance to any source or output
        source_dist = np.linalg.norm(obj_center - source_pos)
        output_dist = np.linalg.norm(obj_center - output_pos)
        min_distance = min(source_dist, output_dist)

        # Determine LOD level based on distance
        percent, aggression = self._calculate_lod_level(min_distance, adr_threshold)

        simplified = mesh.simplify_quadric_decimation(percent=percent, aggression=aggression)

        # Return appropriate LOD mesh
        if simplified.is_watertight and simplified.is_volume and simplified.is_winding_consistent:
            return simplified
        else:
            return mesh

    def _calculate_lod_level(self, distance: float, adr_threshold: float) -> int:
        """
        Calculate LOD level based on distance.
        
        Args:
            distance: Minimum distance to source/output
            adr_threshold: Base threshold for LOD switching
        
        Returns:
            LOD level (0 = highest detail, >1 lowest detail))
        """
        # Calculate aggression factor based on how much threshold is exceeded
        if distance <= adr_threshold:
            return 0, 0 # Full detail
        
        # Calculate how many threshold multiples we're beyond
        threshold_multiples = distance / adr_threshold
        
        # Map to quadric_decimation
        if threshold_multiples < 2:
            percent = threshold_multiples - 1
            aggression = 1
        elif threshold_multiples > 2:
            percent = 1
            aggression = int(threshold_multiples - 1)

        return percent, aggression
