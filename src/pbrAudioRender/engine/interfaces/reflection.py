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

from lib.interpolate import FrequencyInterpolator


class ReflectionInterface:
    """Handle sound wave reflection at material boundaries"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        self.max_reflections = config.acoustic_domain.max_reflections
    
    @nb.jit(nopython=True, parallel=True)
    def apply_reflection(self, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                        reflection_coeffs: np.ndarray, normal_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply reflection at boundaries"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        for i in nb.prange(1, pressure.shape[0]-1):
            for j in range(1, pressure.shape[1]-1):
                for k in range(1, pressure.shape[2]-1):
                    R = reflection_coeffs[i, j, k]
                    
                    if R > 0:  # Reflection occurs
                        normal = normal_vectors[i, j, k]
                        
                        # Simple reflection model
                        # In practice, use more sophisticated reflection models
                        reflection_strength = R
                        
                        # Reflect pressure (simplified)
                        new_pressure[i, j, k] += pressure[i, j, k] * reflection_strength
                        
                        # Reflect velocity components based on surface normal
                        dot_product = (vx[i, j, k] * normal[0] + 
                                     vy[i, j, k] * normal[1] + 
                                     vz[i, j, k] * normal[2])
                        
                        new_vx[i, j, k] -= 2 * dot_product * normal[0] * reflection_strength
                        new_vy[i, j, k] -= 2 * dot_product * normal[1] * reflection_strength
                        new_vz[i, j, k] -= 2 * dot_product * normal[2] * reflection_strength
        
        return new_pressure, new_vx, new_vy, new_vz
    
    def _calculate_surface_normals(self, soxel_grid) -> np.ndarray:
        """Calculate surface normals from soxel types"""
        # Simplified normal calculation
        # In practice, use gradient of impedance or material properties
        normals = np.zeros(soxel_grid.shape + (3,), dtype=np.float32)
        
        for i in range(1, soxel_grid.shape[0]-1):
            for j in range(1, soxel_grid.shape[1]-1):
                for k in range(1, soxel_grid.shape[2]-1):
                    # Simple gradient-based normal calculation
                    if soxel_grid.soxel_types[i, j, k] != soxel_grid.soxel_types[i+1, j, k]:
                        normals[i, j, k] = np.array([1.0, 0.0, 0.0])
                    elif soxel_grid.soxel_types[i, j, k] != soxel_grid.soxel_types[i, j+1, k]:
                        normals[i, j, k] = np.array([0.0, 1.0, 0.0])
                    elif soxel_grid.soxel_types[i, j, k] != soxel_grid.soxel_types[i, j, k+1]:
                        normals[i, j, k] = np.array([0.0, 0.0, 1.0])
        
        return normals
    
    def update_step(self, layer_manager, soxel_grid, frequency: float = 1000.0):
        """Apply reflection to fields"""
        if not self.config.interface.reflection_enabled:
            return layer_manager
        
        # Get reflection coefficients for current frequency
        reflection_coeffs = soxel_grid.get_acoustic_property_grid("reflection", frequency)
        
        # Calculate surface normals
        surface_normals = self._calculate_surface_normals(soxel_grid)
        
        # Apply reflection
        new_pressure, new_vx, new_vy, new_vz = self.apply_reflection(
            layer_manager.pressure,
            layer_manager.velocity_x,
            layer_manager.velocity_y,
            layer_manager.velocity_z,
            reflection_coeffs,
            surface_normals
        )
        
        # Update layer manager
        layer_manager.pressure = new_pressure
        layer_manager.velocity_x = new_vx
        layer_manager.velocity_y = new_vy
        layer_manager.velocity_z = new_vz
        
        return layer_manager

