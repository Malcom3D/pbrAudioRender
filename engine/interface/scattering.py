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
Scattering handling at material interfaces and rough surfaces.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import numba as nb

from ...utils.parallel_proc import configure_numba


class ScatteringManager:
    """Manages sound wave scattering at material interfaces"""
    
    def __init__(self, config, gpu_manager=None):
        self.config = config
        self.gpu = gpu_manager
        self.jit = configure_numba(parallel=True)
        self.rng = np.random.default_rng()
    
    def apply_scattering(self, fields: Dict[str, np.ndarray],
                        boundaries: List[Dict[str, Any]],
                        soxel_grid) -> Dict[str, np.ndarray]:
        """
        Apply scattering at detected boundaries.
        
        Args:
            fields: Current acoustic fields
            boundaries: Detected boundary information
            soxel_grid: SoxelGrid for material properties
        
        Returns:
            Fields with scattering applied
        """
        if not boundaries:
            return fields
        
        result_fields = fields.copy()
        
        if self.gpu and self.gpu.config.use_gpu:
            result_fields = self._apply_scattering_gpu(result_fields, boundaries, soxel_grid)
        else:
            result_fields = self._apply_scattering_cpu(result_fields, boundaries, soxel_grid)
        
        return result_fields
    
    def _apply_scattering_cpu(self, fields: Dict[str, np.ndarray],
                            boundaries: List[Dict[str, Any]],
                            soxel_grid) -> Dict[str, np.ndarray]:
        """CPU implementation of scattering"""
        pressure = fields['pressure'].copy()
        vx = fields['velocity_x'].copy()
        vy = fields['velocity_y'].copy()
        vz = fields['velocity_z'].copy()
        
        shape = pressure.shape
        
        for boundary in boundaries:
            i, j, k = boundary['position']
            
            current_soxel = boundary['current_soxel']
            
            # Get scattering coefficient
            avg_frequency = 1000.0  # Placeholder
            scattering_coeff = current_soxel.get_property_at_frequency(
                current_soxel.scattering_coeffs, avg_frequency
            )
            
            if scattering_coeff > 0:
                # Apply scattering by redistributing energy to neighboring voxels
                scattered_energy = pressure[i, j, k] * scattering_coeff
                
                # Reduce energy at current voxel
                pressure[i, j, k] *= (1 - scattering_coeff)
                
                # Distribute scattered energy to neighbors
                neighbors = self._get_scattering_neighbors((i, j, k), shape)
                energy_per_neighbor = scattered_energy / len(neighbors)
                
                for ni, nj, nk in neighbors:
                    pressure[ni, nj, nk] += energy_per_neighbor
        
        return {
            'pressure': pressure,
            'velocity_x': vx,
            'velocity_y': vy,
            'velocity_z': vz
        }
    
    def _get_scattering_neighbors(self, position: Tuple[int, int, int],
                                shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get valid neighboring positions for scattering"""
        i, j, k = position
        neighbors = []
        
        # Check all 26 neighbors in 3x3x3 cube (excluding center)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if di == 0 and dj == 0 and dk == 0:
                        continue  # Skip center
                    
                    ni, nj, nk = i + di, j + dj, k + dk
                    
                    if (0 <= ni < shape[0] and 
                        0 <= nj < shape[1] and 
                        0 <= nk < shape[2]):
                        neighbors.append((ni, nj, nk))
        
        return neighbors
    
    def _apply_scattering_gpu(self, fields: Dict[str, np.ndarray],
                            boundaries: List[Dict[str, Any]],
                            soxel_grid) -> Dict[str, np.ndarray]:
        """GPU implementation of scattering"""
        # For now, fall back to CPU implementation
        return self._apply_scattering_cpu(fields, boundaries, soxel_grid)

