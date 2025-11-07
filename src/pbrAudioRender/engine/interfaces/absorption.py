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

from ...lib.base import Configurable, GPUEnabled
from ...lib.interpolate import FrequencyInterpolator


class AbsorptionInterface(Configurable, GPUEnabled):
    """Handle sound energy absorption at material boundaries"""
    
    def __init__(self, config=None, gpu_manager=None):
        super().__init__(config)
        GPUEnabled.__init__(self, gpu_manager)
    
    @nb.jit(nopython=True, parallel=True)
    def apply_absorption(self, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                        absorption_coeffs: np.ndarray, frequency: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply frequency-dependent absorption to fields"""
        new_pressure = np.zeros_like(pressure)
        new_vx = np.zeros_like(vx)
        new_vy = np.zeros_like(vy)
        new_vz = np.zeros_like(vz)
        
        for i in nb.prange(pressure.shape[0]):
            for j in range(pressure.shape[1]):
                for k in range(pressure.shape[2]):
                    alpha = absorption_coeffs[i, j, k]
                    
                    # Apply absorption (simplified model)
                    absorption_factor = 1.0 - alpha
                    
                    new_pressure[i, j, k] = pressure[i, j, k] * absorption_factor
                    new_vx[i, j, k] = vx[i, j, k] * absorption_factor
                    new_vy[i, j, k] = vy[i, j, k] * absorption_factor
                    new_vz[i, j, k] = vz[i, j, k] * absorption_factor
        
        return new_pressure, new_vx, new_vy, new_vz
    
    def update_step(self, layer_manager, soxel_grid, frequency: float = 1000.0):
        """Apply absorption to fields"""
        if not self.config.interface.absorption_enabled:
            return layer_manager
        
        # Get absorption coefficients for current frequency
        absorption_coeffs = soxel_grid.get_acoustic_property_grid("absorption", frequency)
        
        # Apply absorption
        new_pressure, new_vx, new_vy, new_vz = self.apply_absorption(
            layer_manager.pressure,
            layer_manager.velocity_x,
            layer_manager.velocity_y,
            layer_manager.velocity_z,
            absorption_coeffs,
            frequency
        )
        
        # Update layer manager
        layer_manager.pressure = new_pressure
        layer_manager.velocity_x = new_vx
        layer_manager.velocity_y = new_vy
        layer_manager.velocity_z = new_vz
        
        return layer_manager

