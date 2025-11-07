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
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numba as nb

from core.entity_manager import EntityManager

@dataclass
class BoundaryConditions:
    """Handle boundary conditions for the simulation domain"""
    entity_manager: EntityManager
    idx: int
    
    def __post_init__(self):
        pass

#        self.boundary_type = getattr(config.wave_propagation, 'boundary_type', 'open')
#        self.absorption_coeff = getattr(config.wave_propagation, 'boundary_absorption', 0.95)
    
    @nb.jit(nopython=True, parallel=True)
    def apply_open_boundary(self, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                          absorption: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply open (absorbing) boundary conditions"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        # Apply absorption at domain boundaries
        for i in nb.prange(pressure.shape[0]):
            for j in range(pressure.shape[1]):
                for k in range(pressure.shape[2]):
                    # Check if we're near a boundary
                    on_boundary = (
                        i == 0 or i == pressure.shape[0] - 1 or
                        j == 0 or j == pressure.shape[1] - 1 or
                        k == 0 or k == pressure.shape[2] - 1
                    )
                    
                    if on_boundary:
                        # Calculate distance to nearest boundary (normalized)
                        dist_x = min(i, pressure.shape[0] - 1 - i) / pressure.shape[0]
                        dist_y = min(j, pressure.shape[1] - 1 - j) / pressure.shape[1]
                        dist_z = min(k, pressure.shape[2] - 1 - k) / pressure.shape[2]
                        
                        min_dist = min(dist_x, dist_y, dist_z)
                        
                        # Apply stronger absorption closer to boundary
                        boundary_absorption = absorption * (1.0 - min_dist * pressure.shape[0] / 2.0)
                        boundary_absorption = max(0.0, min(1.0, boundary_absorption))
                        
                        absorption_factor = 1.0 - boundary_absorption
                        
                        new_pressure[i, j, k] *= absorption_factor
                        new_vx[i, j, k] *= absorption_factor
                        new_vy[i, j, k] *= absorption_factor
                        new_vz[i, j, k] *= absorption_factor
        
        return new_pressure, new_vx, new_vy, new_vz
    
    @nb.jit(nopython=True, parallel=True)
    def apply_periodic_boundary(self, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply periodic boundary conditions"""
        # For periodic boundaries, we need to wrap around the edges
        # This is handled in the FDTD solver, so here we just pass through
        return pressure, vx, vy, vz
    
    def update_step(self, layer_manager, soxel_grid):
        """Apply boundary conditions"""
        if self.boundary_type == 'open':
            new_pressure, new_vx, new_vy, new_vz = self.apply_open_boundary(
                layer_manager.pressure,
                layer_manager.velocity_x,
                layer_manager.velocity_y,
                layer_manager.velocity_z,
                self.absorption_coeff
            )
        elif self.boundary_type == 'periodic':
            new_pressure, new_vx, new_vy, new_vz = self.apply_periodic_boundary(
                layer_manager.pressure,
                layer_manager.velocity_x,
                layer_manager.velocity_y,
                layer_manager.velocity_z
            )
        else:  # rigid (reflective)
            # Rigid boundaries are handled by reflection interface
            return layer_manager
        
        # Update layer manager
        layer_manager.pressure = new_pressure
        layer_manager.velocity_x = new_vx
        layer_manager.velocity_y = new_vy
        layer_manager.velocity_z = new_vz
        
        return layer_manager

