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
Refraction handling at material interfaces using Snell's Law.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import numba as nb

from ...utils.parallel_proc import configure_numba


class RefractionManager:
    """Manages sound wave refraction at material interfaces"""
    
    def __init__(self, config, gpu_manager=None):
        self.config = config
        self.gpu = gpu_manager
        self.jit = configure_numba(parallel=True)
    
    def apply_refraction(self, fields: Dict[str, np.ndarray],
                        boundaries: List[Dict[str, Any]],
                        soxel_grid) -> Dict[str, np.ndarray]:
        """
        Apply refraction at detected boundaries using Snell's Law.
        
        Args:
            fields: Current acoustic fields
            boundaries: Detected boundary information
            soxel_grid: SoxelGrid for material properties
        
        Returns:
            Fields with refraction applied
        """
        if not boundaries:
            return fields
        
        result_fields = fields.copy()
        
        if self.gpu and self.gpu.config.use_gpu:
            result_fields = self._apply_refraction_gpu(result_fields, boundaries, soxel_grid)
        else:
            result_fields = self._apply_refraction_cpu(result_fields, boundaries, soxel_grid)
        
        return result_fields
    
    def _apply_refraction_cpu(self, fields: Dict[str, np.ndarray],
                            boundaries: List[Dict[str, Any]],
                            soxel_grid) -> Dict[str, np.ndarray]:
        """CPU implementation of refraction"""
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
            
            # Calculate refraction using Snell's Law
            c1 = current_soxel.sound_speed
            c2 = neighbor_soxel.sound_speed
            
            # Incident angle (simplified - using boundary normal)
            # In practice, you'd calculate the actual wave direction
            incident_angle = self._calculate_incident_angle(
                (di, dj, dk), (vx[i,j,k], vy[i,j,k], vz[i,j,k])
            )
            
            # Snell's Law: sin(θ1)/c1 = sin(θ2)/c2
            if c2 > 0:
                sin_theta2 = (c2 / c1) * np.sin(incident_angle)
                sin_theta2 = np.clip(sin_theta2, -1.0, 1.0)
                transmission_angle = np.arcsin(sin_theta2)
            else:
                transmission_angle = incident_angle
            
            # Calculate transmission coefficient
            transmission_coeff = self._calculate_transmission_coefficient(
                current_soxel, neighbor_soxel, incident_angle, transmission_angle
            )
            
            # Apply refraction by adjusting pressure and velocity
            pressure[ni, nj, nk] *= transmission_coeff
            vx[ni, nj, nk] *= transmission_coeff
            vy[ni, nj, nk] *= transmission_coeff
            vz[ni, nj, nk] *= transmission_coeff
        
        return {
            'pressure': pressure,
            'velocity_x': vx,
            'velocity_y': vy,
            'velocity_z': vz
        }
    
    def _calculate_incident_angle(self, normal: Tuple[float, float, float],
                                velocity: Tuple[float, float, float]) -> float:
        """Calculate incident angle between wave direction and boundary normal"""
        normal_norm = np.linalg.norm(normal)
        velocity_norm = np.linalg.norm(velocity)
        
        if normal_norm == 0 or velocity_norm == 0:
            return 0.0
        
        dot_product = np.dot(normal, velocity)
        cos_angle = dot_product / (normal_norm * velocity_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return np.arccos(cos_angle)
    
    def _calculate_transmission_coefficient(self, soxel1, soxel2,
                                          incident_angle: float,
                                          transmission_angle: float) -> float:
        """Calculate transmission coefficient for sound waves"""
        z1 = soxel1.density * soxel1.sound_speed
        z2 = soxel2.density * soxel2.sound_speed
        
        if z2 == 0:
            return 0.0
        
        # Normal incidence transmission coefficient
        T_normal = 2 * z2 / (z1 + z2)
        
        # Adjust for oblique incidence
        T_oblique = T_normal * np.cos(transmission_angle) / np.cos(incident_angle)
        
        return np.clip(T_oblique, 0.0, 1.0)
    
    def _apply_refraction_gpu(self, fields: Dict[str, np.ndarray],
                            boundaries: List[Dict[str, Any]],
                            soxel_grid) -> Dict[str, np.ndarray]:
        """GPU implementation of refraction"""
        # For now, fall back to CPU implementation
        return self._apply_refraction_cpu(fields, boundaries, soxel_grid)

