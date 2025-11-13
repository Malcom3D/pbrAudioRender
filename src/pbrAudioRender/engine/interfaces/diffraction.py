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
class DiffractionInterface:
    """Handle sound wave diffraction around obstacles using UTD model"""
    entity_manager: EntityManager
    idx: int
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.enable_diffraction = config.interface.enable_diffraction
    
    @nb.jit(nopython=True)
    def utd_diffraction_coefficient(self, incident_angle: float, diffraction_angle: float, 
                                  frequency: float, obstacle_size: float) -> complex:
        """Calculate UTD diffraction coefficient"""
        k = 2 * np.pi * frequency / 343.0  # wave number
        
        # Edge diffraction parameters
        L = obstacle_size
        n = 2.0  # Wedge angle parameter (n=2 for 90° wedge)
        
        # UTD diffraction coefficient (simplified)
        beta = incident_angle
        beta_prime = diffraction_angle
        
        # Fresnel integrals approximation
        F = 1.0 / (1j * np.sqrt(2 * np.pi * k * L))
        
        # Diffraction coefficient
        D = F * (1.0 / (n * np.sin(np.pi/n))) * (
            1.0 / (np.cos(np.pi/n) - np.cos((beta - beta_prime)/n)) +
            1.0 / (np.cos(np.pi/n) - np.cos((beta + beta_prime)/n))
        )
        
        return D
    
    @nb.jit(nopython=True, parallel=True)
    def apply_diffraction(self, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                         soxel_types: np.ndarray, boundaries: Dict, frequency: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply diffraction to fields at edge boundaries"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        wavelength = 343.0 / frequency if frequency > 0 else 1.0
        
        # Process edge boundaries
        for boundary_idx in nb.prange(len(boundaries.get('edge_boundaries', []))):
            boundary = boundaries['edge_boundaries'][boundary_idx]
            i, j, k = boundary['position']
            
            edge_pressure = pressure[i, j, k]
            
            if np.abs(edge_pressure) > 1e-6:
                # Simplified diffraction - spread energy to shadow region
                diffraction_strength = wavelength / 0.1  # Scale with wavelength
                
                # Distribute pressure to neighboring voxels in shadow region
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            ni, nj, nk = i + di, j + dj, k + dk
                            
                            if (0 <= ni < pressure.shape[0] and 
                                0 <= nj < pressure.shape[1] and 
                                0 <= nk < pressure.shape[2]):
                                
                                # Check if neighbor is in shadow (not object)
                                if soxel_types[ni, nj, nk] != 2:
                                    distance = np.sqrt(di**2 + dj**2 + dk**2)
                                    if distance > 0:
                                        # Apply inverse square law with diffraction
                                        diffracted_pressure = (
                                            edge_pressure * diffraction_strength / 
                                            (distance * distance)
                                        )
                                        new_pressure[ni, nj, nk] += diffracted_pressure
        
        return new_pressureressure, new_vx, new_vy, new_vz
    
    def update_step(self, layer_manager, soxel_grid, boundaries: Dict, frequency: float = 1000.0):
        """Apply diffraction to fields"""
        if not self.enable_diffraction:
            return layer_manager
        
        # Apply diffraction
        new_pressure, new_vx, new_vy, new_vz = self.apply_diffraction(
            layer_manager.get_array('fdtd', 0, 'pressure'),
            layer_manager.get_array('fdtd', 0, 'vx'),
            layer_manager.get_array('fdtd', 0, 'vy'),
            layer_manager.get_array('fdtd', 0, 'vz'),
            soxel_grid.soxel_types,
            boundaries,
            frequency
        )
        
        # Update layer manager
        wave_propagator = self.entity_manager.get('wave_propagators', self.idx)
        layer_manager = wave_propagator.layer_manager
        
        return layer_manager
