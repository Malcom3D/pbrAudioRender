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


class ScatteringInterface:
    """Handle sound wave scattering at rough surfaces"""
    
    def __init__(self, config=None):
        super().__init__(config)
    
    @nb.jit(nopython=True, parallel=True)
    def apply_scattering(self, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                        scattering_coeffs: np.ndarray, frequency: float, wavelength: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply scattering to fields"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        # Calculate scattering parameters based on frequency/wavelength
        scattering_strength = frequency / 1000.0  # Simplified frequency dependence
        
        for i in nb.prange(1, pressure.shape[0]-1):
            for j in range(1, pressure.shape[1]-1):
                for k in range(11, pressure.shape[2]-1):
                    scatter_coeff = scattering_coeffs[i, j, k]
                    
                    if scatter_coeff > 0:
                        # Apply diffuse scattering
                        # Convert some specular energy to diffuse
                        diffuse_fraction = scatter_coeff * scattering_strength
                        
                        # Simple scattering model - redistribute energy randomly
                        random_phase = np.exp(1j * 2 * np.pi * np.random.random())
                        
                        new_pressure[i, j, k] *= (1 - diffuse_fraction) + diffuse_fraction * random_phase.real
                        
                        # Scatter velocity components
                        scatter_dir = np.array([
                            np.random.random() - 0.5,
                            np.random.random() - 0.5, 
                            np.random.random() - 0.5
                        ])
                        scatter_dir = scatter_dir / np.linalg.norm(scatter_dir)
                        
                        velocity_mag = np.sqrt(vx[i,j,k]**2 + vy[i,j,k]**2 + vz[i,j,k]**2)
                        scattered_velocity = scatter_dir * velocity_mag * diffuse_fraction
                        
                        new_vx[i, j, k] = vx[i, j, k] * (1 - diffuse_fraction) + scattered_velocity[0]
                        new_vy[i, j, k] = vy[i, j, k] * (1 - diffuse_fraction) + scattered_velocity[1]
                        new_vz[i, j, k] = vz[i, j, k] * (1 - diffuse_fraction) + scattered_velocity[2]
        
        return new_pressure, new_vx, new_vy, new_vz
    
    def update_step(self, layer_manager, soxel_grid, frequency: float = 1000.0):
        """Apply scattering to fields"""
        if not self.config.interface.scattering_enabled:
            return layer_manager
        
        # Get scattering coefficients for current frequency
        scattering_coeffs = soxel_grid.get_acoustic_property_grid("scattering", frequency)
        
        # Calculate wavelength
        sound_speed = soxel_grid.get_acoustic_property_grid("sound_speed", frequency)
        avg_sound_speed = np.mean(sound_speed)
        wavelength = avg_sound_speed / frequency if frequency > 0 else 1.0
        
        # Apply scattering
        new_pressure, new_vx, new_vy, new_vz = self.apply_scattering(
            layer_manager.pressure,
            layer_manager.velocity_x,
            layer_manager.velocity_y,
            layer_manager.velocity_z,
            scattering_coeffs,
            frequency,
            wavelength
        )
        
        # Update layer manager
        layer_manager.pressure = new_pressure
        layer_manager.velocity_x = new_vx
        layer_manager.velocity_y = new_vy
        layer_manager.velocity_z = new_vz
        
        return layer_manager

