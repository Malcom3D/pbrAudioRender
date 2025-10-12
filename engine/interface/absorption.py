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
Absorption handling at material interfaces.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import numba as nb

from ...utils.parallel_proc import configure_numba


class AbsorptionManager:
    """Manages sound energy absorption at material interfaces"""
    
    def __init__(self, config, gpu_manager=None):
        self.config = config
        self.gpu = gpu_manager
        self.jit = configure_numba(parallel=True)
    
    def apply_absorption(self, fields: Dict[str, np.ndarray], 
                        boundaries: List[Dict[str, Any]],
                        soxel_grid) -> Dict[str, np.ndarray]:
        """
        Apply absorption at detected boundaries.
        
        Args:
            fields: Current acoustic fields
            boundaries: Detected boundary information
            soxel_grid: SoxelGrid for material properties
        
        Returns:
            Fields with absorption applied
        """
        if not boundaries:
            return fields
        
        result_fields = fields.copy()
        
        if self.gpu and self.gpu.config.use_gpu:
            result_fields = self._apply_absorption_gpu(result_fields, boundaries, soxel_grid)
        else:
            result_fields = self._apply_absorption_cpu(result_fields, boundaries, soxel_grid)
        
        return result_fields
    
    def _apply_absorption_cpu(self, fields: Dict[str, np.ndarray],
                            boundaries: List[Dict[str, Any]],
                            soxel_grid) -> Dict[str, np.ndarray]:
        """CPU implementation of absorption"""
        pressure = fields['pressure'].copy()
        
        for boundary in boundaries:
            i, j, k = boundary['position']
            ni, nj, nk = boundary['neighbor_position']
            
            # Get absorption coefficients
            current_soxel = boundary['current_soxel']
            neighbor_soxel = boundary['neighbor_soxel']
            
            # Use average frequency for absorption calculation
            avg_frequency = 1000.0  # Placeholder - should come from FDTD
            
            abs_current = current_soxel.get_property_at_frequency(
                current_soxel.absorption_coeffs, avg_frequency
            )
            abs_neighbor = neighbor_soxel.get_property_at_frequency(
                neighbor_soxel.absorption_coeffs, avg_frequency
            )
            
            # Average absorption coefficient
            absorption = (abs_current + abs_neighbor) / 2.0
            
            # Apply absorption to pressure field
            # Simple exponential decay model
            decay_factor = np.exp(-absorption)
            pressure[i, j, k] *= decay_factor
            pressure[ni, nj, nk] *= decay_factor
        
        fields['pressure'] = pressure
        return fields
    
    def _apply_absorption_gpu(self, fields: Dict[str, np.ndarray],
                            boundaries: List[Dict[str, Any]],
                            soxel_grid) -> Dict[str, np.ndarray]:
        """GPU implementation of absorption"""
        # For now, fall back to CPU implementation
        return self._apply_absorption_cpu(fields, boundaries, soxel_grid)

