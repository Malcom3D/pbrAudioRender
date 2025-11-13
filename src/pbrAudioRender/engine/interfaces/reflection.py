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
from dataclasses import dataclass, field

from core.entity_manager import EntityManager
from lib.interpolate import FrequencyInterpolator

@dataclass
class ReflectionInterface:
    """Handle sound wave reflection at material boundaries with layer management"""
    entity_manager: EntityManager
    idx: int
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.enable_reflection = config.interface.enable_reflection
    
    @nb.jit(nopython=True, parallel=True)
    def apply_reflection(self, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                        reflection_coeffs: np.ndarray, boundaries: Dict, normal_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply reflection at boundaries"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        for boundary_idx in nb.prange(len(boundaries['impedance_discontinuities'])):
            boundary = boundaries['impedance_discontinuities'][boundary_idx]
            i, j, k k = boundary['position']
            
            R = reflection_coeffs[i, j, k]
            
            if R > 0:  # Reflection occurs
                normal = normal_vectors[i, j, k]
                
                # Reflection strength based on impedance ratio
                reflection_strength = R
                
                # Reflect pressure
                new_pressure[i, j, k] += pressure[i, j, k] * reflection_strength
                
                # Reflect velocity components based on surface normal
                dot_product = (vx[i, j, k] * normal[0] + 
                             vy[i, j, k] * normal[1] + 
                             vz[i, j, k] * normal[2])
                
                new_vx[i, j, k] -= 2 * dot_product * normal[0] * reflection_strength
                new_vy[i, j, k] -= 2 * dot_product * normal[1] * reflection_strength
                new_vz[i, j, k] -= 2 * dot_product * normal[2] * reflection_strength
        
        return new_pressure, new_vx, new_vy, new_vz
    
    def _calculate_surface_normals(self, soxel_grid, boundaries: Dict) -> np.ndarray:
        """Calculate surface normals from boundaries"""
        normals = np.zeros(soxel_grid.shape + (3,), dtype=np.float32)
        
        for boundary in boundaries['impedance_discontinuities']:
            i, j, k = boundary['position']
            ni, nj, nk = boundary['neighbor_position']
            
            # Normal points from current voxel to neighbor
            normal = np.array([ni-i, nj-j, nk-k], dtype=np.float32)
            normal_length = np.linalg.norm(normal)
            
            if normal_length > 0:
                normals[i, j, k] = normal / normal_length
        
        return normals
    
    def update_step(self, layer_manager, soxel_grid, boundaries: Dict, frequency: float = 1000.0):
        """Apply reflection to fields"""
        if not self.enable_reflection:
            return layer_manager
        
        # Get reflection coefficients for current frequency
        reflection_coeffs = soxel_grid.get_acoustic_property_grid("reflection", frequency)
        
        # Calculate surface normals from boundaries
        surface_normals = self._calculate_surface_normals(soxel_grid, boundaries)
        
        # Apply reflection
        new_pressure, new_vx, new_vy, new_vz = self.apply_reflection(
            layer_manager.get_array('fdtd', 0, 'pressure'),
            layer_manager.get_array('fdtd', 0, 'vx'),
            layer_manager.get_array('fdtd', 0, 'vy'),
            layer_manager.get_array('fdtd', 0, 'vz'),
            reflection_coeffs,
            boundaries,
            surface_normals
        )
        
        # Update layer manager
        wave_propagator = self.entity_manager.get('wave_propagators', self.idx)
        layer_manager = wave_propagator.layer_manager
        
        return layer_manager

