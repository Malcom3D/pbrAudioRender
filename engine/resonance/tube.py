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
Tube resonance detection and simulation.
Handles resonances in cylindrical, conical, and rectangular tubes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import numba as nb
from scipy import ndimage

from ...utils.parallel_proc import configure_numba


class TubeResonator:
    """Detects and simulates tube resonances"""
    
    def __init__(self, config, gpu_manager=None):
        self.config = config
        self.gpu = gpu_manager
        self.jit = configure_numba(parallel=True)
    
    def detect(self, soxel_grid) -> List[Dict[str, Any]]:
        """
        Detect Detect tube-like structures in the scene.
        
        Args:
            soxel_grid: Current SoxelGrid state
        
        Returns:
            List of detected tube resonators
        """
        resonators = []
        shape = soxel_grid.shape
        
        # Find elongated cavities (potential tubes)
        elongated_cavities = self._find_elongated_cavities(soxel_grid)
        
        for cavity in elongated_cavities:
            tube_info = self._analyze_tube_geometry(cavity, soxel_grid)
            
            if tube_info:
                resonance_freqs = self._calculate_tube_frequencies(tube_info, soxel_grid)
                
                if resonance_freqs:
                    resonator = {
                        'type': 'tube',
                        'tube_voxels': cavity,
                        'tube_info': tube_info,
                        'resonance_frequencies': resonance_freqs,
                        'end_conditions': self._determine_end_conditions(cavity, soxel_grid)
                    }
                    resonators.append(resonator)
        
        return resonators
    
    def apply_resonance(self, fields: Dict[str, np.ndarray],
                       resonances: List[Dict[str, Any]],
                       soxel_grid) -> Dict[str, np.ndarray]:
        """
        Apply tube resonance effects to acoustic fields.
        
        Args:
            fields: Current acoustic fields
            resonances: Detected resonances
            soxel_grid: Current SoxelGrid state
        
        Returns:
            Fields with tube resonances applied
        """
        result_fields = fields.copy()
        tube_resonances = [r for r in resonances if r['type'] == 'tube']
        
        if not tube_resonances:
            return result_fields
        
        if self.gpu and self.gpu.config.use_gpu:
            result_fields = self._apply_tube_gpu(result_fields, tube_resonances, soxel_grid)
        else:
            result_fields = self._apply_tube_cpu(result_fields, tube_resonances, soxel_grid)
        
        return result_fields
    
    def _apply_tube_cpu(self, fields: Dict[str, np.ndarray],
                       resonances: List[Dict[str, Any]],
                       soxel_grid) -> Dict[str, np.ndarray]:
        """CPU implementation of tube resonance"""
        pressure = fields['pressure'].copy()
        
        for resonator in resonances:
            tube_voxels = resonator['tube_voxels']
            resonance_freqs = resonator['resonance_frequencies']
            tube_info = resonator['tube_info']
            end_conditions = resonator['end_conditions']
            
            for freq in resonance_freqs[:3]:  # Apply first 3 modes
                resonance_strength = self._calculate_tube_resonance_strength(
                    resonator, freq, fields, soxel_grid
                )
                
                for i, j, k in tube_voxels:
                    # Tube standing wave pattern
                    standing_wave = self._calculate_tube_standing_wave(
                        (i, j, k), tube_info, end_conditions, freq, soxel_grid
                    )
                    pressure[i, j, k] += resonance_strength * standing_wave
        
        fields['pressure'] = pressure
        return fields
    
    def _find_elongated_cavities(self, soxel_grid) -> List[List[Tuple[int, int, int]]]:
        """Find elongated cavities that could be tubes"""
        shape = soxel_grid.shape
        elongated_cavities = []
        
        # Create air mask (default medium)
        air_mask = np.zeros(shape, dtype=bool)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    soxel = soxel_grid.grid[i, j, k]
                    air_mask[i, j, k] = (soxel.idx == 0)  # Default medium
        
        # Use connected components to find air regions
        structure = np.ones((3, 3, 3), dtype=bool)
        labeled, num_features = ndimage.label(air_mask, structure=structure)
        
        for label in range(1, num_features + 1):
            cavity_voxels = np.argwhere(labeled == label)
            
            # Check if cavity is elongated (tube-like)
            if self._is_cavity_elongated(cavity_voxels):
                elongated_cavities.append([tuple(voxel) for voxel in cavity_voxels])
        
        return elongated_cavities
    
    def _is_cavity_elongated(self, cavity_voxels: np.ndarray, 
                           elongation_threshold: float = 2.0) -> bool:
        """Check if a cavity is elongated (aspect ratio > threshold)"""
        if len(cavity_voxels) < 2:
            return False
        
        positions = np.array(cavity_voxels)
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        dimensions = max_pos - min_pos + 1
        
        # Calculate aspect ratios
        sorted_dims = np.sort(dimensions)
        aspect_ratio = sorted_dims[2] / sorted_dims[0]  # max/min dimension
        
        return aspect_ratio >= elongation_threshold
    
    def _analyze_tube_geometry(self, cavity_voxels: List[Tuple[int, int, int]],
                              soxel_grid) -> Optional[Dict[str, Any]]:
        """Analyze tube geometry and characteristics"""
        if not cavity_voxels:
            return None
        
        positions = np.array(cavity_voxels)
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        dimensions = (max_pos - min_pos) * soxel_grid.voxel_size
        
        # Find principal axis (longest dimension)
        principal_axis = np.argmax(dimensions)
        tube_length = dimensions[principal_axis]
        
        # Calculate cross-sectional area
        cross_section_dims = [dim for i, dim in enumerate(dimensions) if i != principal_axis]
        if len(cross_section_dims) == 2:
            cross_section_area = cross_section_dims[0] * cross_section_dims[1]
        else else:
            cross_section_area = soxel_grid.voxel_size ** 2  # Fallback
        
        # Determine tube shape
        shape_ratio = max(cross_section_dims) / min(cross_section_dims) if min(cross_section_dims) > 0 else 1.0
        if shape_ratio < 1.5:
            tube_shape = "cylindrical"
        elif shape_ratio > 3.0:
            tube_shape = "rectangular"
        else:
            tube_shape = "unknown"
        
        return {
            'principal_axis': principal_axis,
            'length': tube_length,
            'cross_section_area': cross_section_area,
            'shape': tube_shape,
            'min_pos': min_pos,
            'max_pos': max_pos
        }
    
    def _calculate_tube_frequencies(self, tube_info: Dict[str, Any],
                                  soxel_grid) -> List[float]:
        """Calculate resonance frequencies for tube"""
        length = tube_info['length']
        
        if length <= 0:
            return []
        
        # Get average sound speed in tube
        tube_voxels = []
        min_pos = tube_info['min_pos']
        max_pos = tube_info['max_pos']
        axis = tube_info['principal_axis']
        
        # Reconstruct tube vox voxels (simplified)
        for i in range(min_pos[0], max_pos[0] + 1):
            for j in range(min_pos[1], max_pos[1] + 1):
                for k in range(min_pos[2], max_pos[2] + 1):
                    if (i, j, k) in tube_voxels:  # This would need proper reconstruction
                        tube_voxels.append((i, j, k))
        
        if not tube_voxels:
            return []
        
        sound_speeds = []
        for i, j, k in tube_voxels:
            soxel = soxel_grid.grid[i, j, k]
            sound_speeds.append(soxel.sound_speed)
        
        avg_sound_speed = np.mean(sound_speeds)
        
        # Calculate resonance frequencies based on end conditions
        # For open-open tube: f = n * c / (2 * L)
        # For closed-closed tube: f = n * c / (2 * L)  
        # For open-closed tube: f = (2n-1) * c / (4 * L)
        frequencies = []
        
        # Assume open-open for now (simplified)
        for n in range(1, 6):  # First 5 modes
            freq = n * avg_sound_speed / (2 * length)
            frequencies.append(freq)
        
        return frequencies
    
    def _determine_end_conditions(self, tube_voxels: List[Tuple[int, int, int]],
                                soxel_grid) -> Dict[str, str]:
        """Determine if tube ends are open or closed"""
        # Simplified implementation
        # In practice, check if ends are connected to open space or solid walls
        
        return {
            'end1': 'open',  # Placeholder
            'end2': 'open'   # Placeholder
        }
    
    def _calculate_tube_resonance_strength(self, resonator: Dict[str, Any],
                                         frequency: float,
                                         fields: Dict[str, np.ndarray],
                                         soxel_grid) -> float:
        """Calculate resonance strength for tube"""
        tube_voxels = resonator['tube_voxels']
        
        if not tube_voxels:
            return 0.0
        
        # Calculate average pressure in tube
        avg_pressure = 0.0
        for i, j, k in tube_voxels:
            avg_pressure += fields['pressure'][i, j, k]
        avg_pressure /= len(tube_voxels)
        
        # Strength depends on tube dimensions and excitation
        tube_volume = len(tube_voxels) * (soxel_grid.voxel_size ** 3)
        base_strength = avg_pressure / max(tube_volume, 1e-6)
        
        return base_strength
    
    def _calculate_tube_standing_wave(self, position: Tuple[int, int, int],
                                    tube_info: Dict[str, Any],
                                    end_conditions: Dict[str, str],
                                    frequency: float,
                                    soxel_grid) -> float:
        """Calculate standing wave pattern in tube at position"""
        axis = tube_info['principal_axis']
        length = tube_info['length']
        min_pos = tube_info['min_pos']
        
        # Normalized position along tube (0 to 1)
        tube_pos = position[axis] - min_pos[axis]
        normalized_pos = tube_pos * soxel_grid.voxel_size / length
        
        # Standing wave pattern based on end conditions
        if end_conditions['end1'] == 'open' and end_conditions['end2'] == 'open':
            # Open-open: sin(n * π * x / L)
            mode_number = int(round(frequency * 2 * length / 343.0))
            standing_wave = np.sin(mode_number * np.pi * normalized_pos)
        elif end_conditions['end1'] == 'closed' and end_conditions['end2'] == 'closed':
            # Closed-closed: cos(n * π * x / L)  
            mode_number = int(round(frequency * 2 * length / 343.0))
            standing_wave = np.cos(mode_number * np.pi * normalized_pos)
        else:
            # Open-closed: sin((2n-1) * π * x / (2L))
            mode_number = int(round(frequency * 4 * length / 343.0))
            standing_wave = np.sin((2 * mode_number - 1) * np.pi * normalized_pos / 2)
        
        # Time oscillation
        time_oscillation = np.sin(2 * np.pi * frequency * soxel_grid.current_time)
        
        return standing_wave * time_oscillation
    
    def _apply_tube_gpu(self, fields: Dict[str, np.ndarray],
                       resonances: List[Dict[str, Any]],
                       soxel_grid) -> Dict[str, np.ndarray]:
        """GPU implementation of tube resonance"""
        # For now, fall back to CPU implementation
        return self._apply_tube_cpu(fields, resonances, soxel_grid)


