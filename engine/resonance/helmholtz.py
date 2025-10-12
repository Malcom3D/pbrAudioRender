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
Helmholtz resonator detection and simulation.
Handles resonances in cavities with necked openings.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import numba as nb
from scipy import ndimage

from ...utils.parallel_proc import configure_numba


class HelmholtzResonator:
    """Detects and simulates Helmholtz resonators"""
    
    def __init__(self, config, gpu_manager=None):
        self.config = config
        self.gpu = gpu_manager
        self.jit = configure_numba(parallel=True)
    
    def detect(self, soxel_grid) -> List[Dict[str, Any]]:
        """
        Detect Helmholtz resonators in the scene.
        
        Args:
            soxel_grid: Current SoxelGrid state
        
        Returns:
            List of detected Helmholtz resonators
        """
        resonators = []
        shape = soxel_grid.shape
        
        # Find enclosed cavities (regions completely surrounded by solid/object voxels)
        cavities = self._find_enclosed_cavities(soxel_grid)
        
        for cavity in cavities:
            # Check if cavity has necked openings (characteristic of Helmholtz resonators)
            neck_info = self._find_cavity_necks(cavity, soxel_grid)
            
            if neck_info:
                resonator = {
                    'type': 'helmholtz',
                    'cavity_voxels': cavity,
                    'neck_voxels': neck_info['neck_voxels'],
                    'neck_area': neck_info['neck_area'],
                    'cavity_volume': len(cavity) * (soxel_grid.voxel_size ** 3),
                    'resonance_frequency': self._calculate_helmholtz_frequency(
                        cavity, neck_info, soxel_grid
                    )
                }
                resonators.append(resonator)
        
        return resonators
    
    def apply_resonance(self, fields: Dict[str, np.ndarray],
                       resonances: List[Dict[str, Any]],
                       soxel_grid) -> Dict[str, np.ndarray]:
        """
        Apply Helmholtz resonance effects to acoustic fields.
        
        Args:
            fields: Current acoustic fields
            resonances: Detected resonances
            soxel_grid: Current SoxelGrid state
        
        Returns:
            Fields with Helmholtz resonances applied
        """
        result_fields = fields.copy()
        helmholtz_resonances = [r for r in resonances if r['type'] == 'helmholtz']
        
        if not helmholtz_resonances:
            return result_fields
        
        if self.gpu and self.gpu.config.use_gpu:
            result_fields = self._apply_helmholtz_gpu(result_fields, helmholtz_resonances, soxel_grid)
        else:
            result_fields = self._apply_helmholtz_cpu(result_fields, helmholtz_resonances, soxel_grid)
        
        return result_fields
    
    def _apply_helmholtz_cpu(self, fields: Dict[str, np.ndarray],
                           resonances: List[Dict[str, Any]],
                           soxel_grid) -> Dict[str, np.ndarray]:
        """CPU implementation of Helmholtz resonance"""
        pressure = fields['pressure'].copy()
        
        for resonator in resonances:
            cavity_voxels = resonator['cavity_voxels']
            resonance_freq = resonator['resonance_frequency']
            neck_voxels = resonator['neck_voxels']
            
            # Calculate resonance excitation from neck
            neck_excitation = self._calculate_neck_excitation(
                fields, neck_voxels, soxel_grid
            )
            
            # Apply resonance to cavity
            resonance_strength = self._calculate_resonance_strength(
                resonator, neck_excitation, soxel_grid
            )
            
            for i, j, k in cavity_voxels:
                # Helmholtz resonance adds oscillating pressure
                time_factor = np.sin(2 * np.pi * resonance_freq * soxel_grid.current_time)
                pressure[i, j, k] += resonance_strength * time_factor
        
        fields['pressure'] = pressure
        return fields
    
    def _find_enclosed_cavities(self, soxel_grid) -> List[List[Tuple[int, int, int]]]:
        """Find completely enclosed cavities in the voxel grid"""
        shape = soxel_grid.shape
        visited = np.zeros(shape, dtype=bool)
        cavities = []
        
        # Create solid mask (non-default medium)
        solid_mask = np.zeros(shape, dtype=bool)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    soxel = soxel_grid.grid[i, j, k]
                    solid_mask[i, j, k] = (soxel.idx !=  0)  # Not default medium
        
        # Use connected components to find enclosed regions
        structure = np.ones((3, 3, 3), dtype=bool)
        labeled, num_features = ndimage.label(~solid_mask, structure=structure)
        
        for label in range(1, num_features + 1):
            cavity_voxels = np.argwhere(labeled == label)
            
            # Check if cavity is enclosed (all neighbors are solid)
            if self._is_cavity_enclosed(cavity_voxels, solid_mask, shape):
                cavities.append([tuple(voxel) for voxel in cavity_voxels])
        
        return cavities
    
    def _is_cavity_enclosed(self, cavity_voxels: np.ndarray, 
                          solid_mask: np.ndarray, shape: Tuple[int, int, int]) -> bool:
        """Check if a cavity is completely enclosed by solid voxels"""
        for voxel in cavity_voxels:
            i, j, k = voxel
            
            # Check all 6 direct neighbors
            for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                ni, nj, nk = i + di, j + dj, k + dk
                
                # If neighbor is outside grid or not solid, cavity is not enclosed
                if (ni < 0 or ni >= shape[0] or 
                    nj < 0 or nj >= shape[1] or 
                    nk < 0 or nk >= shape[2] or 
                    not solid_mask[ni, nj, nk]):
                    return False
        
        return True
    
    def _find_cavity_necks(self, cavity_voxels: List[Tuple[int, int, int]],
                          soxel_grid) -> Optional[Dict[str, Any]]:
        """Find neck-like openings in a cavity"""
        shape = soxel_grid.shape
        neck_voxels = []
        
        # Look for voxels that connect the cavity to the outside
        for i, j, k in cavity_voxels:
            # Check if this voxel has an external connection
            external_connections = 0
            for di, dj, dk in [(1,0,0), (- (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                ni, nj, nk = i + di, j + dj, k + dk
                
                if (0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]):
                    neighbor_soxel = soxel_grid.grid[ni, nj, nk]
                    if neighbor_soxel.idx == 0:  # Default medium (air)
                        external_connections += 1
            
            # Consider it a neck voxel if it has limited external connections
            if 0 < external_connections <= 2:
                neck_voxels.append((i, j, k))
        
        if neck_voxels:
            neck_area = len(neck_voxels) * (soxel_grid.voxel_size ** 2)
            return {
                'neck_voxels': neck_voxels,
                'neck_area': neck_area
            }
        
        return None
    
    def _calculate_helmholtz_frequency(self, cavity_voxels: List[Tuple[int, int, int]],
                                     neck_info: Dict[str, Any],
                                     soxel_grid) -> float:
        """Calculate Helmholtz resonance frequency"""
        # Helmholtz frequency: f = (c/2π) * √(A/(V*L))
        # where c = sound speed, A = neck area, V = cavity volume, L = neck length
        
        cavity_volume = len(cavity_voxels) * (soxel_grid.voxel_size ** 3)
        neck_area = neck_info['neck_area']
        
        # Estimate neck length (assume 1 voxel for simplicity)
        neck_length = soxel_grid.voxel_size
        
        # Get average sound speed in cavity
        sound_speeds = []
        for i, j, k in cavity_voxels:
            soxel = soxel_grid.grid[i, j, k]
            sound_speeds.append(soxel.sound_speed)
        
        avg_sound_speed = np.mean(sound_speeds)
        
        if cavity_volume > 0 and neck_length > 0:
            frequency = (avg_sound_speed / (2 * np.pi)) * np.sqrt(
                neck_area / (cavity_volume * neck_length)
            )
            return float(frequency)
        
        return 0.0
    
    def _calculate_neck_excitation(self, fields: Dict[str, np.ndarray],
                                 neck_voxels: List[Tuple[int, int, int]],
                                 soxel_grid) -> float:
        """Calculate excitation at resonator neck"""
        if not neck_voxels:
            return 0.0
        
        total_pressure = 0.0
        for i, j, k in neck_voxels:
            total_pressure += fields['pressure'][i, j, k]
        
        return total_pressure / len(neck_voxels)
    
    def _calculate_resonance_strength(self, resonator: Dict[str, Any],
                                    neck_excitation: float,
                                    soxel_grid) -> float:
        """Calculate resonance strength based on geometry and excitation"""
        cavity_volume = resonator['cavity_volume']
        neck_area = resonator['neck_area']
        
        # Strength proportional to neck area and inverse to cavity volume
        base_strength = neck_area / max(cavity_volume, 1e-6)
        
        return base_strength * neck_excitation
    
    def _apply_helmholtz_gpu(self, fields: Dict[str, np.ndarray],
                           resonances: List[Dict[str, Any]],
                           soxel_grid) -> Dict[str, np.ndarray]:
        """GPU implementation of Helmholtz resonance"""
        # For now, fall back to CPU implementation
        return self._apply_helmholtz_cpu(fields, resonances, soxel_grid)

