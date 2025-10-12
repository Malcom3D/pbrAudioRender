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
Reflection handling at material interfaces.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import numba as nb

from ...utils.parallel_proc import configure_numba


class ReflectionManager:
    """Manages sound wave reflection at material interfaces"""
    
    def __init__(self, config, gpu_manager=None):
        self.config = config
        self.gpu = gpu_manager
        self.jit = configure_numba(parallel=True)
    
    def apply_reflection(self, fields: Dict[str, np.ndarray],
                        boundaries: List[Dict[str, Any]],
                        soxel_grid) -> Dict[str, np.ndarray]:
        """
        Apply reflection at detected boundaries.
        
        Args:
            fields: Current acoustic fields
            boundaries: Detected boundary information
            soxel_grid: SoxelGrid for material properties
        
        Returns:
            Fields with reflection applied
        """
        if not boundaries:
            return fields
        
        result_fields = fields.copy()
        
        if self.gpu and self.gpu.config.use_gpu:
            result_fields = self._apply_reflection_gpu(result_fields, boundaries, soxel_grid)
        else:
            result_fields = self._apply_reflection_cpu(result_fields, boundaries, soxel_grid)
        
        return result_fields
    
    def _apply_reflection_cpu(self, fields: Dict[str, np.ndarray],
                            boundaries: List[Dict[str, Any]],
                            soxel_grid) -> Dict[str, np.ndarray]:
        """CPU implementation of reflection"""
        pressure = fields['pressure'].copy()
        vx = fields['velocity_x'].copy()
        vy = fields['velocity_y'].copy()
        vz = fields['velocity_z'].copy()
        
        for boundary in boundaries:
            i, j, k = boundary['position']
            ni, nj, nk = boundary['neighbor_position']
            di, dj, dk = boundary['direction']
            
            current_soxel = boundary['current_soxel']
            neighbor_soxel = boundary['neighbor_soxel']
            
            # Calculate reflection coefficient
            reflection_coeff = self._calculate_reflection_coefficient(
                current_soxel, neighbor_soxel
            )
            
            # Apply reflection by reversing velocity component normal to boundary
            # and scaling by reflection coefficient
            vx[i, j, k] -= 2 * vx[i, j, k] * di * reflection_coeff
            vy[i, j, k] -= 2 * vy[i, j, k] * dj * reflection_coeff
            vz[i, j, k] -= 2 * vz[i, j, k] * dk * reflection_coeff
            
            # Also reflect pressure field
            pressure[i, j, k] *= reflection_coeff
        
        return {
            'pressure': pressure,
            'velocity_x': vx,
            'velocity_y': vy,
            'velocity_z': vz
        }
    
    def _calculate_reflection_coefficient(self, soxel1, soxel2) -> float:
        """Calculate reflection coefficient based on impedance mismatch"""
        z1 = soxel1.density * soxel1.sound_speed
        z2 = soxel2.density * soxel2.sound_speed
        
        if z1 + z2 == 0:
            return 1.0  # Total reflection
        
        R = (z2 - z1) / (z2 + z1)
        
        # Get frequency-dependent reflection coefficient
        avg_frequency = 1000.0  # Placeholder
        freq_reflection = soxel1.get_property_at_frequency(
            soxel1.reflection_coeffs, avg_frequency
        )
        
        # Combine impedance-based and frequency-based reflection
        combined_reflection = abs(R) * freq_reflection
        
        return np.clip(combined_reflection, 0.0, 1.0)
    
    def _apply_reflection_gpu(self, fields: Dict[str, np.ndarray],
                            boundaries: List[Dict[str, Any]],
                            soxel_grid) -> Dict[str, np.ndarray]:
        """GPU implementation of reflection"""
        # For now, fall back to CPU implementation
        return self._apply_reflection_cpu(fields, boundaries, soxel_grid)

