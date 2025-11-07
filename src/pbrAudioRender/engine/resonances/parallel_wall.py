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

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import numba as nb

from ...lib.base import Configurable, GPUEnabled


class ParallelWallResonance(Configurable, GPUEnabled):
    """Handle standing wave resonances between parallel walls"""
    
    def __init__(self, config=None, gpu_manager=None):
        super().__init__(config)
        GPUEnabled.__init__(self, gpu_manager)
    
    def detect(self, soxel_grid) -> List[Dict]:
        """Detect parallel wall structures in the scene"""
        wall_pairs = []
        
        # Look for parallel opposing walls
        for axis in [0, 1, 2]:  # x, y, z axes
            walls = self._find_walls_on_axis(soxel_grid, axis)
            wall_pairs.extend(self._find_parallel_wall_pairs(walls, axis))
        
        return wall_pairs
    
    def _find_walls_on_axis(self, soxel_grid, axis: int) -> List[Dict]:
        """Find wall-like structures along given axis"""
        walls = []
        
        if axis == 0:  # x-axis walls
            for j in range(soxel_grid.shape[1]):
                for k in range(soxel_grid.shape[2]):
                    wall = self._find_wall_in_column(soxel_grid, axis, j, k)
                    if wall:
                        walls.append(wall)
        
        elif axis == 1:  # y-axis walls  
            for i in range(soxel_grid.shape[0]):
                for k in range(soxel_grid.shape[2]):
                    wall = self._find_wall_in_column(soxel_grid, axis, i, k)
                    if wall:
                        walls.append(wall)
        
        else:  # z-axis walls
            for i in range(soxel_grid.shape[0]):
                for j in range(soxel_grid.shape[1]):
                    wall = self._find_wall_in_column(soxel_grid, axis, i, j)
                    if wall:
                        walls.append(wall)
        
        return walls
    
    def _find_wall_in_column(self, soxel_grid, axis: int, coord1: int, coord2: int) -> Optional[Dict]:
        """Find wall in a column along given axis"""
        if axis == 0:
            column = soxel_grid.soxel_types[:, coord1, coord2]
        elif axis == 1:
            column = soxel_grid.soxel_types[coord1, :, coord2]
        else:
            column = soxel_grid.soxel_types[coord1, coord2, :]
        
        # Find continuous object segments
        object_segments = []
        in_segment = False
        start_idx = 0
        
        for idx, voxel_type in enumerate(column):
            if voxel_type == 2 and not in_segment:  # Start of object segment
                in_segment = True
                start_idx = idx
            elif voxel_type != 2 and in_segment:  # End of object segment
                in_segment = False
                object_segments.append((start_idx, idx-1))
        
        if in_segment:
            object_segments.append((start_idx, len(column)-1))
        
        # Find wall-like segments (long continuous objects)
        for start, end in object_segments:
            length = end - start + 1
            if length >= 3:  # Minimum wall length
                return {
                    'axis': axis,
                    'coord1': coord1,
                    'coord2': coord2,
                    'start': start,
                    'end': end,
                    'position': (coord1, coord2) if axis != 0 else (start, coord1, coord2)
                }
        
        return None
    
    def _find_parallel_wall_pairs(self, walls: List[Dict], axis: int) -> List[Dict]:
        """Find pairs of parallel opposing walls"""
        pairs = []
        
        # Group walls by their coordinates
        wall_groups = {}
        for wall in walls:
            key = (wall['coord1'], wall['coord2'])
            if key not in wall_groups:
                wall_groups[key] = []
            wall_groups[key].append(wall)
        
        # Find opposing walls in each group
        for key, wall_list in wall_groups.items():
            if len(wall_list) >= 2:
                # Sort by position along axis
                wall_list.sort(key=lambda w: w['start'])
                
                # Find adjacent wall pairs
                for i in range(len(wall_list)-1):
                    wall1 = wall_list[i]
                    wall2 = wall_list[i+1]
                    
                    # Check if they form a cavity
                    cavity_start = wall1['end'] + 1
                    cavity_end = wall2['start'] - 1
                    cavity_length = cavity_end - cavity_start + 1
                    
                    if cavity_length >= 2:  # Minimum cavity size
                        pairs.append({
                            'axis': axis,
                            'wall1': wall1,
                            'wall2': wall2,
                            'cavity_start': cavity_start,
                            'cavity_end': cavity_end,
                            'cavity_length': cavity_length * soxel_grid.voxel_size,
                            'position': key
                        })
        
        return pairs
    
    def calculate_resonance_frequency(self, wall_pair: Dict) -> List[float]:
        """Calculate standing wave resonance frequencies between parallel walls"""
        # f_n = (n * c) / (2 * L) for n = 1, 2, 3, ...
        c = 343.0  # sound speed
        L = wall_pair['cavity_length']
        
        if L <= 0:
            return []
        
        frequencies = []
        max_modes = self.config.resonance.max_resonance_modes
        
        for n in range(1, max_modes + 1):
            f = (n * c) / (2 * L)
            if f <= 20000:  # Audible range
                frequencies.append(f)
        
        return frequencies
    
    @nb.jit(nopython=True, parallel=True)
    def apply_parallel_wall_resonance(self, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                                     wall_pairs: List[Dict], resonance_freqs: List[List[float]],
                                     current_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply parallel wall resonance effects"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        for pair_idx in nb.prange(len(wall_pairs)):
            wall_pair = wall_pairs[pair_idx]  # Note: numba doesn't handle dicts well
            freqs = resonance_freqs[pair_idx]
            
            # Simplified implementation - in practice, use proper mode shapes
            for f in freqs:
                if f > 0:
                    # Apply standing wave pattern in cavity
                    axis = wall_pair['axis']
                    cavity_start = wall_pair['cavity_start']
                    cavity_end = wall_pair['cavity_end']
                    
                    # Create standing wave pattern
                    for pos in range(cavity_start, cavity_end + 1):
                        if axis == 0:
                            x = pos
                            # Standing wave in x-direction
                            wave_amplitude = np.sin(np.pi * (pos - cavity_start) / (cavity_end - cavity_start))
                            resonant_pressure = pressure[x, wall_pair['coord1'], wall_pair['coord2']] * wave_amplitude
                            new_pressure[x, wall_pair['coord1'], wall_pair['coord2']] += resonant_pressure
                        # Similar for other axes...
        
        return new_pressure, new_vx, new_vy, new_vz
    
    def update(self, layer_manager, soxel_grid, current_time: float = 0.0):
        """Apply parallel wall resonance effects"""
        if not self.config.resonance.enable_parallel_wall:
            return layer_manager
        
        # Detect wall pairs
        wall_pairs = self.detect(soxel_grid)
        
        if not wall_pairs:
            return layer_manager
        
        # Calculate resonance frequencies
        resonance_freqs = [self.calculate_resonance_frequency(pair) for pair in wall_pairs]
        
        # Apply resonance
        new_pressure, new_vx, new_vy, new_vz = self.apply_parallel_wall_resonance(
            layer_manager.pressure,
            layer_manager.velocity_x,
            layer_manager.velocity_y,
            layer_manager.velocity_z,
            wall_pairs,
            resonance_freqs,
            current_time
        )
        
        # Update layer manager
        layer_manager.pressure = new_pressure
        layer_manager.velocity_x = new_vx
        layer_manager.velocity_y = new_vy
        layer_manager.velocity_z = new_vz
        
        return layer_manager

