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


class RefractionInterface:
    """Handle sound wave refraction at material boundaries"""
    
    def __init__(self, config=None):
        super().__init__(config)
    
    @nb.jit(nopython=True)
    def snells_law(self, incident_angle: float, sound_speed1: float, sound_speed2: float) -> float:
        """Calculate refraction angle using Snell's Law"""
        # Snell's Law: sin(θ1)/c1 = sin(θ2)/c2
        sin_thetatheta2 = (sound_speed2 / sound_speed1) * np.sin(incident_angle)
        
        # Handle total internal reflection
        if abs(sin_theta2) > 1.0:
            return np.pi / 2  # Total internal reflection
        
        return np.arcsin(sin_theta2)
    
    @nb.jit(nopython=True, parallel=True)
    def apply_refraction(self, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                        sound_speed: np.ndarray, normal_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply refraction at boundaries"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        for i in nb.prange(1, pressure.shape[0]-1):
            for j in range(1, pressure.shape[1]-1):
                for k in range(1, pressure.shape[2]-1):
                    normal = normal_vectors[i, j, k]
                    
                    # Only process if we have a meaningful normal
                    if np.any(normal != 0):
                        # Get sound speeds (simplified - in practice, use neighboring voxels)
                        c1 = sound_speed[i, j, k]
                        
                        # Estimate incident angle from velocity direction
                        velocity = np.array([vx[i, j, k], vy[i, j, k], vz[i, j, k]])
                        velocity_magnitude = np.sqrt(np.sum(velocity**2))
                        
                        if velocity_magnitude > 1e-6:
                            velocity_dir = velocity / velocity_magnitude
                            incident_angle = np.arccos(np.abs(np.dot(velocity_dir, normal)))
                            
                            # Simple refraction model
                            # In practice, use proper transmission coefficients
                            transmission_strength = 0.8  # Simplified
                            
                            # Apply refraction by modifying velocity direction
                            refracted_velocity = velocity * transmission_strength
                            
                            new_vx[i, j, k] = refracted_velocity[0]
                            new_vy[i, j, k] = refracted_velocity[1]
                            new_vz[i, j, k] = refracted_velocity[2]
        
        return new_pressure, new_vx, new_vy, new_vz
    
    def update_step(self, layer_manager, soxel_grid, frequency: float = 1000.0):
        """Apply refraction to fields"""
        if not self.config.interface.refraction_enabled:
            return layer_manager
        
        # Get sound speed for current frequency
        sound_speed = soxel_grid.get_acoustic_property_grid("sound_speed", frequency)
        
        # Calculate surface normals (reuse from reflection or calculate separately)
        surface_normals = np.zeros(soxel_grid.shape + (3,), dtype=np.float32)
        # In practice, calculate proper normals
        
        # Apply refraction
        new_pressure, new_vx, new_vy, new_vz = self.apply_refraction(
            layer_manager.pressure,
            layer_manager.velocity_x,
            layer_manager.velocity_y,
            layer_manager.velocity_z,
            sound_speed,
            surface_normals
        )
        
        # Update layer manager
        layer_manager.pressure = new_pressure
        layer_manager.velocity_x = new_vx
        layer_manager.velocity_y = new_vy
        layer_manager.velocity_z = new_vz
        
        return layer_manager

