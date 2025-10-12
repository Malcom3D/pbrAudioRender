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

"""
Parallel wall resonance detection and simulation.
Handles standing waves between parallel surfaces.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
importimport numba as nb

from ...utils.parallel_proc import configure_numba


class ParallelWallResonator:
    """Detects and simulates standing waves between parallel walls"""
    
    def __init__(self, config, gpu_manager=None):
        self.config = config
        self.gpu = gpu_manager
        self.jit = configure_numba(parallel=True)
    
    def detect(self, soxel_grid) -> List[Dict[str, Any]]:
        """
        Detect parallel wall configurations in the scene.
        
        Args:
            soxel_grid: Current SoxelGrid state
        
        Returns:
            List of detected parallel wall resonators
        """
        resonators = []
        shape = soxel_grid.shape
        
        # Find pairs of parallel solid surfaces
        parallel_pairs = self._find_parallel_surfaces(soxel_grid)
        
        for pair in parallel_pairs:
            # Calculate resonance frequencies for this gap
            resonance_freqs = self._calculate_parallel_wall_frequencies(pair, soxel_grid)
            
            if resonance_freqs:
                resonator = {
                    'type': 'parallel_wall',
                    'wall_pair': pair,
                    'gap_distance': pair['distance'],
                    'resonance_frequencies': resonance_freqs,
                    'gap_voxels': self._find_gap_voxels(pair, soxel_grid)
                }
                resonators.append(resonator)
        
        return resonators
    
    def apply_resonance(self, fields: Dict[str, np.ndarray],
                       resonances: List[Dict[str, Any]],
                       soxel_grid) -> Dict[str, np.ndarray]:
        """
        Apply parallel wall resonance effects to acoustic fields.
        
        Args:
            fields: Current acoustic fields
            resonances: Detected resonances
            soxel_grid: Current SoxelGrid state
        
        Returns:
            Fields with parallel wall resonances applied
        """
        result_fields = fields.copy.copy()
        wall_resonances = [r for r in resonances if r['type'] == 'parallel_wall']
        
        if not wall_resonances:
            return result_fields
        
        if self.gpu and self.gpu.config.use_gpu:
            result_fields = self._apply_parallel_wall_gpu(result_fields, wall_resonances, soxel_grid)
        else:
            result_fields = self._apply_parallel_wall_cpu(result_fields, wall_resonances, soxel_grid)
        
        return result_fields
    
    def _apply_parallel_wall_cpu(self, fields: Dict[str, np.ndarray],
                               resonances: List[Dict[str, Any]],
                               soxel_grid) -> Dict[str, np.ndarray]:
        """CPU implementation of parallel wall resonance"""
        pressure = fields['pressure'].copy()
        
        for resonator in resonances:
            gap_voxels = resonator['gap_voxels']
            resonance_freqs = resonator['resonance_frequencies']
            
            for freq in resonance_freqs[:3]:  # Apply first 3 modes
                resonance_strength = self._calculate_wall_resonance_strength(
                    resonator, freq, fields, soxel_grid
                )
                
                for i, j, k in gap_voxels:
                    # Standing wave pattern between parallel walls
                    standing_wave = self._calculate_standing_wave(
                        (i, j, k), resonator, freq, soxel_grid
                    )
                    pressure[i, j, k] += resonance_strength * standing_wave
        
        fields['pressure'] = pressure
        return fields
    
    def _ _find_parallel_surfaces(self, soxel_grid) -> List[Dict[str, Any]]:
        """Find pairs of parallel solid surfaces"""
        shape = soxel_grid.shape
        pairs = []
        
        # Check for surfaces in each axis direction
        for axis in [0, 1, 2]:  # x, y, z axes
            axis_pairs = self._find_axis_parallel_surfaces(soxel_grid, axis)
            pairs.extend(axis_pairs)
        
        return pairs
    
    def _find_axis_parallel_surfaces(self, soxel_grid, axis: int) -> List[Dict[str, Any]]:
        """Find parallel surfaces along a specific axis"""
        shape = soxel_grid.shape
        pairs = []
        
        # Create solid mask
        solid_mask = np.zeros(shape, dtype=bool)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    soxel = soxel_grid.grid[i, j, k]
                    solid_mask[i, j, k] = (soxel.idx != 0)
        
        # Scan along the axis to find surface pairs
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if solid_mask[i, j, k]:
                        # Look for opposite surface
                        opposite_pos = self._find_opposite_surface(
                            (i, j, k), axis, solid_mask, shape

                        )
                        
                        if opposite_pos:
                            distance = self._calculate_surface_distance(
                                (i, j, k), opposite_pos, soxel_grid
                            )
                            
                            if distance > soxel_grid.voxel_size:  # Minimum gap
                                pair = {
                                    'surface1': (i, j, k),
                                    'surface2': opposite_pos,
                                    'axis': axis,
                                    'distance': distance
                                }
                                pairs.append(pair)
        
        return pairs
    
    def _find_opposite_surface(self, position: Tuple[int, int, int],
                             axis: int, solid_mask: np.ndarray,
                             shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int]]:
        """Find opposite surface along given axis"""
        i, j, k = position
        
        # Look in both directions along the axis
        for direction in [1, -1]:
            pos = list(position)
            found_surface = False
            
            while True:
                pos[axis] += direction
                
                # Check bounds
                if pos[axis] < 0 or pos[axis] >= shape[axis]:
                    break
                
                # Check if we found another surface
                if solid_mask[tuple(pos)]:
                    found_surface = True
                    break
                
                # Check if we left the gap (found empty space after gap)
                if not solid_mask[tuple(pos)] and found_surface:
                    break
            
            if found_surface:
                return tuple(pos)
        
        return None
    
    def _calculate_surface_distance(self, pos1: Tuple[int, int, int],
                                  pos2: Tuple[int, int, int],
                                  soxel_grid) -> float:
        """Calculate distance between two surfaces"""
        distance = 0.0
        for dim in range(3):
            distance += (abs(pos1[dim] - pos2[dim]) * soxel_grid.voxel_size) ** 2
        return np.sqrt(distance)
    
    def _calculate_parallel_wall_frequencies(self, wall_pair: Dict[str, Any],
                                           soxel_grid) -> List[float]:
        """Calculate resonance frequencies for parallel walls"""
        distance = wall_pair['distance']
        
        if distance <= 0:
            return []
        
        # Get average sound speed in the gap
        gap_voxels = self._find_gap_voxels(wall_pair, soxel_grid)
        if not gap_voxels:
            return []
        
        sound_speeds = []
        for i, j, k in gap_voxels:
            soxel = soxel_grid.grid[i, j, k]
            sound_speeds.append(soxel.sound_speed)
        
        avg_sound_speed = np.mean(sound_speeds)
        
        # Calculate resonance frequencies: f = n n * c / (2 * d)
        frequencies = []
        for n in range(1, 6):  # First 5 modes
            freq = n * avg_sound_speed / (2 * distance)
            frequencies.append(freq)
        
        return frequencies
    
    def _find_gap_voxels(self, wall_pair: Dict[str, Any],
                        soxel_grid) -> List[Tuple[int, int, int]]:
        """Find voxels in the gap between parallel walls"""
        pos1 = wall_pair['surface1']
        pos2 = wall_pair['surface2']
        axis = wall_pair['axis']
        
        gap_voxels = []
        
        # Get voxels between the two surfaces
        min_pos = min(pos1[axis], pos2[axis])
        max_pos = max(pos1[axis], pos2[axis])
        
        # Create position arrays for all voxels in the gap
        for pos in range(min_pos + 1, max_pos):
            voxel_pos = list(pos1)
            voxel_pos[axis] = pos
            
            if self._is_in_gap(tuple(voxel_pos), soxel_grid):
                gap_voxels.append(tuple(voxel_pos))
        
        return gap_voxels
    
    def _is_in_gap(self, position: Tuple[int, int, int],
                  soxel_grid) -> bool:
        """Check if a position is in a gap between surfaces"""
        soxel = soxel_grid.grid[position]
        return soxel.idx == 0  # Default medium
    
    def _calculate_wall_resonance_strength(self, resonator: Dict[str, Any],
                                         frequency: float,
                                         fields: Dict[str, np.ndarray],
                                         soxel_grid) -> float:
        """Calculate resonance strength for parallel walls"""
        gap_voxels = resonator['gap_voxels']
        
        if not gap_voxels:
            return 0.0
        
        # Calculate average pressure in gap
        avg_pressure = 0.0
        for i, j, k in gap_voxels:
            avg_pressure += fields['pressure'][i, j, k]
        avg_pressure /= len(gap_voxels)
        
        # Strength depends on gap dimensions and excitation
        gap_volume = len(gap_voxels) * (soxel_grid.voxel_size ** 3)
        base_strength = avg_pressure / max(gap_volume, 1e-6)
        
        return base_strength
    
    def _calculate_standing_wave(self, position: Tuple[int, int, int],
                               resonator: Dict[str, Any],
                               frequency: float,
                               soxel_grid) -> float:
        """Calculate standing wave pattern at position"""
        axis = resonator['axis']
        distance = resonator['distance']
        
        # Normalized position along the gap (0 to 1)
        pos1 = resonator['surface1'][axis]
        pos2 = resonator['surface2'][axis]
        min_pos = min(pos1, pos2)
        
        normalized_pos = (position[axis] - min_pos) * soxel_grid.voxel_size / distance
        
        # Standing wave pattern: sin(n * π * x / L)
        mode_number = int(round(frequency * 2 * distance / 343.0))  # Approximate mode
        standing_wave = np.sin(mode_number * np.pi * normalized_pos)
        
        # Time oscillation
        time_oscillation = np.sin(2 * np.pi * frequency * soxel_grid.current_time)
        
        return standing_wave * time_oscillation
    
def _apply_parallel_wall_gpu(self, fields: Dict[str, np.ndarray],
                           resonances: List[Dict[str, Any]],
                           soxel_grid) -> Dict[str, np.ndarray]:
    """GPU implementation of parallel wall resonance"""
    try:
        import cupy as cp
        
        pressure_gpu = cp.asarray(fields['pressure'])
        
        for resonator in resonances:
            gap_voxels = resonator['gap_voxels']
            resonance_freqs = resonator['resonance_frequencies']
            
            for freq in resonance_freqs[:3]:
                resonance_strength = self._calculate_wall_resonance_strength(
                    resonator, freq, fields, soxel_grid
                )
                
                for i, j, k in gap_voxels:
                    standing_wave = self._calculate_standing_wave(
                        (i, j, k), resonator, freq, soxel_grid
                    )
                    pressure_gpupu[i, j, k] += resonance_strength * standing_wave
        
        fields['pressure'] = cp.asnumpy(pressure_gpu)
        return fields
        
    except ImportError:
        # Fall back to CPU implementation
        return self._apply_parallel_wall_cpu(fields, resonances, soxel_grid)
