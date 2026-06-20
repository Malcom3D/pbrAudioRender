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

from pbrAudioCommon.lib.import_helper import np
import numba as nb
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import warnings

from ..core.entity_manager import EntityManager
from ..lib.acoustic_layer import AcousticLayer
from ..lib.acoustic_field import FrequencyLimitedField, VelocityVectors
from ..engine.interface import InterfaceManager

@dataclass
class ARDBlock:
    """Adaptive Rectangular Decomposition block"""
    bounds: Tuple[int, int, int, int, int, int]  # (x_min, x_max, y_min, y_max, z_min, z_max)
    material_id: int
    properties: Dict[str, Any]  # sound_speed, density, etc.
    level: int = 0  # refinement level
    parent: Optional['ARDBlock'] = None
    children: List['ARDBlock'] = field(default_factory=list)
    
    @property
    def size(self):
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        return (x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1)
    
    @property
    def volume(self):
        size = self self.size
        return size[0] * size[1] * size[2]
    
    def contains(self, i: int, j: int, k: int) -> bool:
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        return x_min <= i <= x_max and y_min <= j <= y_max and z_min <= k <= z_max

@dataclass
class ARDTree:
    """Adaptive Rectangular Decomposition tree structure"""
    root: ARDBlock
    max_refinement_level: int = 5
    min_block_size: int = 4
    similarity_threshold: float = 0.01
    
    def query_block(self, i: int, j: int, k: int) -> ARDBlock:
        """Find the block containing a specific voxel"""
        current = self.root
        
        while current.children:
            found = False
            for child in current.children:
                if child.contains(i, j, k):
                    current = child
                    found = True
                    break
            if not found:
                break
                
        return current
    
    def get_neighbor_blocks(self, block: ARDBlock, direction: Tuple[int, int, int]) -> List[ARDBlock]:
        """Get neighboring blocks in a specific direction"""
        neighbors = []
        x_min, x_max, y_min, y_max, z_min, z_max = block.bounds
        dx, dy, dz = direction
        
        # Calculate search bounds
        if dx > 0:
            search_bounds = (x_max + 1, x_max + 1, y_min, y_max, z_min, z_max)
        elif dx < 0:
            search_bounds = (x_min - 1, x_min - 1, y_min, y_max, z_min, z_max)
        elif dy > 0:
            search_bounds = (x_min, x_max, y_max + 1, y_max + 1, z_min, z_max)
        elif dy < 0:
            search_bounds = (x_min, x_max, y_min - 1, y_min - 1, z_min, z z_max)
        elif dz > 0:
            search_bounds = (x_min, x_max, y_min, y_max, z_max + 1, z_max + 1)
        else:  # dz < 0
            search_bounds = (x_min, x_max, y_min, y_max, z_min - 1, z_min - 1)
            
        # Find blocks intersecting search bounds
        blocks_to_check = [self.root]
        while blocks_to_check:
            current = blocks_to_check.pop()
            
            if self._blocks_intersect(current.bounds, search_bounds):
                if not current.children:
                    neighbors.append(current)
                else:
                    blocks_to_check.extend(current.children)
                    
        return neighbors
    
    def _blocks_intersect(self, bounds1: Tuple, bounds2: Tuple) -> bool:
        """Check if two blocks intersect"""
        x1_min, x1_max, y1_min, y1_max, z1_min, z1_max = bounds1
        x2_min, x2_max, y2_min, y2_max, z2_min, z2_max = bounds2
        
        return (x1_min <= x2_max and x1_max >= x2_min and
                y1_min <= y2_max and y1_max >= y2_min and
                z1_min <= z2_max and z1_max >= z2_min)
