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
Damping manager for preventing energy accumulation in the simulation.
Implements various damping models for numerical stability and physical accuracy.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import numba as nb

from ..utils.parallel_proc import configure_numba
from ..utils.gpu_acceleration import GPUManager


class DampingManager:
    """Manages damping operations to prevent energy accumulation"""
    
    def __init__(self, config, gpu_manager: Optional[GPUManager] = None):
        self.config = config
        self.gpu = gpu_manager
        self.jit = configure_numba(parallel=True)
        
        # Damping parameters
        self.damping_coefficient = 0.01
        self.spatial_damping = True
        self.temporal_damping = True
        
    def update_step(self, fdtd_fields: Dict[str, np.ndarray],
                   resonance_fields: Dict[str, np.ndarray],
                   soxel_grid) -> Dict[str, np.ndarray]:
        """
        Apply damping to prevent energy accumulation.
        
        Args:
            fdtd_fields: Fields from FDTD solver
            resonance_fields: Fields after resonance effects
            soxel_grid: Current SoxelGrid state
        
        Returns:
            Damped fields
        """
        result_fields = resonance_fields.copy()
        
        if self.gpu and self.gpu.config.use_gpu:
            result_fields = self._apply_damping_gpu(result_fields, soxel_grid)
        else:
            result_fields = self._apply_damping_cpu(result_fields, soxel_grid)
        
        return result_fields
    
    def _apply_damping_cpu(self, fields: Dict[str, np.ndarray],
                          soxel_grid) -> Dict[str, np.ndarray]:
        """CPU implementation of damping"""
        pressure = fields['pressure'].copy()
        vx = fields['velocity_x'].copy()
        vy = fields['velocity_y'].copy()
        vz = fields['velocity_z'].copy()
        
        shape = pressure.shape
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    soxel = soxel_grid.grid[i, j, k]
                    
                    # Get material-specific damping
                    material_damping = soxel.get_property_at_frequency(
                        soxel.absorption_coeffs, 1000.0  # Reference frequency
                    )
                    
                    # Combine with numerical damping
                    total_damping = self.damping_co_coefficient + material_damping
                    
                    # Apply damping
                    damping_factor = 1.0 - total_damping
                    pressure[i, j, k] *= damping_factor
                    vx[i, j, k] *= damping_factor
                    vy[i, j, k] *= damping_factor
                    vz[i, j, k] *= damping_factor
        
        return {
            'pressure': pressure,
            'velocity_x': vx,
            'velocity_y': vy,
            'velocity_z': vz
        }
    
    def _apply_damping_gpu(self, fields: Dict[str, np.ndarray],
                          soxel_grid) -> Dict[str, np.ndarray]:
        """GPU implementation of damping"""
        # For now, fall back to CPU implementation
        return self._apply_damping_cpu(fields, soxel_grid)
    
    def apply_boundary_damping(self, fields: Dict[str, np.ndarray],
                              boundary_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply enhanced damping at simulation boundaries.
        
        Args:
            fields: Acoustic fields
            boundary_mask: Boolean mask for boundary regions
        
        Returns:
            Fields with boundary damping applied
        """
        result_fields = fields.copy()
        boundary_damping = 0.1  # Stronger damping at boundaries
        
        for field_name in ['pressure', 'velocity_x', 'velocity_y', 'velocity_z']:
            field = result_fields[field_name]
            field[boundary_mask] *= (1.0 - boundary_damping)
            result_fields[field_name] = field
        
        return result_fields
    
    def calculate_energy_decay(self, fields: Dict[str, np.ndarray]) -> float:
        """
        Calculate current energy decay rate.
        
        Args:
            fields: Acoustic fields
        
        Returns:
            Energy decay rate in dB/s
        """
        total_energy = self._calculate_total_energy(fields)
        
        # Simplified decay rate calculation
        decay_rate = total_energy * self.damping_coefficient
        
        return decay_rate
    
    def _calculate_total_energy(self, fields: Dict[str, np.ndarray]) -> float:
        """Calculate total acoustic energy in fields"""
        pressure = fields['pressure']
        vx = fields['velocity_x']
        vy = fields['velocity_y']
        vz = fields['velocity_z']
        
        # Acoustic energy density: 0.5 * (p²/ρc² + ρv²)
        sound_speed = 343.0  # Reference value
        density = 1.2        # Reference value
        
        pressure_energy = np.sum(pressure ** 2) / (density * sound_speed ** 2)
        velocity_energy = density * np.sum(vx ** 2 + vy ** 2 + vz ** 2)
        
        total_energy = 0.5 * (pressure_energy + velocity_energy)
        
        return total_energy

