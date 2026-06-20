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

from pbrAudioCommon.lib.import_helper import np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numba as nb

from ..core.entity_manager import EntityManager

@dataclass
class Damping:
    """Handle energy damping to prevent accumulation and instability"""
    entity_manager: EntityManager
    idx: int
    
    def __post_init__(self):
        pass
        
#        self.damping_coefficient = getattr(config.wave_propagation, 'damping_coefficient', 0.01)
#        self.energy_threshold = getattr(config.wave_propagation, 'energy_threshold', 1e6)
    
    @nb.jit(nopython=True, parallel=True)
    def apply_damping(self, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                     damping_coeff: float, energy_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply damping to acoustic fields"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        # Calculate local energy (simplified)
        for i in nb.prange(pressure.shape[0]):
            for j in range(pressure.shape[1]):
                for k in range(pressure.shape[2]):
                    local_energy = (pressure[i, j, k] ** 2 + 
                                  vx[i, j, k] ** 2 + 
                                  vy[i, j, k] ** 2 + 
                                  vz[i, j, k] ** 2)
                    
                    if local_energy > energy_threshold:
                        # Apply stronger damping to high-energy regions
                        damping_factor = 1.0 - (damping_coeff * np.sqrt(local_energy / energy_threshold))
                    else:
                        damping_factor = 1.0 - damping_coeff
                    
                    damping_factor = max(0.0, min(1.0, damping_factor))
                    
                    new_pressure[i, j, k] *= damping_factor
                    new_vx[i, j, k] *= damping_factor
                    new_vy[i, j, k] *= damping_factor
                    new_vz[i, j, k] *= damping_factor
        
        return new_pressure, new_vx, new_vy, new_vz
    
    def update_step(self, layer_manager, soxel_grid):
        """Apply damping to prevent energy accumulation"""
        if not hasattr(self.config.wave_propagation, 'enable_damping') or not self.config.wave_propagation.enable_damping:
            return layer_manager
        
        # Get current energy
        fields = {
            'pressure': layer_manager.pressure,
            'velocity_x': layer_manager.velocity_x,
            'velocity_y': layer_manager.velocity_y,
            'velocity_z': layer_manager.velocity_z
        }
        current_energy = calculate_acoustic_energy(fields)
        
        # Adjust damping based on total energy
        adaptive_damping = self.damping_coefficient * min(1.0, current_energy / self.energy_threshold)
        
        # Apply damping
        new_pressure, new_vx, new_vy, new_vz = self.apply_damping(
            layer_manager.pressure,
            layer_manager.velocity_x,
            layer_manager.velocity_y,
            layer_manager.velocity_z,
            adaptive_damping,
            self.energy_threshold
        )
        
        # Update layer manager
        layer_manager.pressure = new_pressure
        layer_manager.velocity_x = new_vx
        layer_manager.velocity_y = new_vy
        layer_manager.velocity_z = new_vz
        
        return layer_manager

