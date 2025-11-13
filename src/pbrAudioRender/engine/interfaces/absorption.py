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
class AbsorptionInterface:
    """Handle sound energy absorption at material boundaries with frequency dependence"""
    entity_manager: EntityManager
    idx: int
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.enable_absorption = config.interface.enable_absorption
    
    @nb.j.jit(nopython=True, parallel=True)
    def apply_absorption(self, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                        absorption_coeffs: np.ndarray, boundaries: Dict, frequency: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply frequency-dependent absorption to fields at boundaries"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        # Frequency-dependent absorption scaling (higher frequencies absorbed more)
        freq_scale = min(1.0, frequency / 1000.0)  # Scale with frequency up to 1kHz
        
        for boundary_idx in nb.prange(len(boundaries['impedance_discontinuities'])):
            boundary = boundaries['impedance_discontinuities'][boundary_idx]
            i, j, k = boundary['position']
            
            alpha = absorption_coeffs[i, j, k]
            
            # Apply frequency-scaled absorption
            absorption_factor = 1.0 - (alpha * freq_scale)
            absorption_factor = max(0.0, min(1.0, absorption_factor))
            
            new_pressure[i, j, k] *= absorption_factor
            new_vx[i, j, k] *= absorption_factor
            new_vy[i, j, k] *= absorption_factor
            new_vz[i, j, k] *= absorption_factor
        
        return new_pressure, new_vx, new_vy, new_vz
    
    def update_step(self, layer_manager, soxel_grid, boundaries: Dict, frequency: float = 1000.0):
        """Apply absorption to fields at boundaries"""
        if not self.enable_absorption:
            return layer_manager
        
        # Get absorption coefficients for current frequency
        absorption_coeffs = soxel_grid.get_acoustic_property_grid("absorption", frequency)
        
        # Apply absorption
        new_pressure, new_vx, new_vy, new_vz = self.apply_absorption(
            layer_manager.get_array('fdtd', 0, 'pressure'),
            layer_manager.get_array('fdtd', 0, 'vx'),
            layer_manager.get_array('fdtd', 0, 'vy'),
            layer_manager.get_array('fdtd', 0, 'vz'),
            absorption_coeffs,
            boundaries,
            frequency
        )
        
        # Update layer manager (adapt to your specific update mechanism)
        wave_propagator = self.entity_manager.get('wave_propagators', self.idx)
        layer_manager = wave_propagator.layer_manager
        
        return layer_manager
