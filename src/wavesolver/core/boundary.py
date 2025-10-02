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
import numba
from typing import Tuple, Optional

class BoundaryConditions:
    """Handles various boundary conditions for the acoustic wave solver."""
    
    def __init__(self, dimensions: Tuple[int, int, int], boundary_type: str = "absorbing"):
        self.dimensions = dimensions
        self.boundary_type = boundary_type
        self.boundary_strength = 0.1
        
        # PML (Perfectly Matched Layer) parameters
        self.pml_thickness = 10
        self.pml_max_absorption = 2.0
        
    def apply_boundary_conditions(self, pressure: np.ndarray, velocity: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Apply boundary conditions to pressure and velocity fields."""
        if self.boundary_type == "absorbing":
            self._apply_absorbing_boundaries(pressure, velocity)
        elif self.boundary_type == "pml":
            self._apply_pml_boundaries(pressure, velocity)
        elif self.boundary_type == "rigid":
            self._apply_rigid_boundaries(pressure, velocity)
        elif self.boundary_type == "periodic":
            self._apply_periodic_boundaries(pressure, velocity)
    
    def _apply_absorbing_boundaries(self, pressure: np.ndarray, 
                                  velocity: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Apply simple absorbing boundary conditions."""
        vx, vy, vz = = velocity
        
        # Simple first-order absorbing boundaries
        boundary_strength = self.boundary_strength
        
        # X boundaries
        pressure[0, :, :] *= (1 - boundary_strength)
        pressure[-1, :, :] *= (1 - boundary_strength)
        vx[0, :, :] *= (1 - boundary_strength)
        vx[-1, :, :] *= (1 - boundary_strength)
        
        # Y boundaries
        pressure[:, 0, :] *= (1 - boundary_strength)
        pressure[:, -1, :] *= (1 - boundary_strength)
        vy[:, 0, :] *= (1 - boundary_strength)
        vy[:, -1, :] *= (1 - boundary_strength)
        
        # Z boundaries
        pressure[:, :, 0] *= (1 - boundary_strength)
        pressure[:, :, -1] *= (1 - boundary_strength)
        vz[:, :, 0] *= (1 - boundary_strength)
        vz[:, :, -1] *= (1 - boundary_strength)
    
    def _apply_pml_boundaries(self, pressure: np.ndarray,
                            velocity: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Apply Perfectly Matched Layer boundary conditions."""
        # PML implementation would go here
        # This is a simplified placeholder
        self._apply_absorbing_boundaries(pressure, velocity)
    
    def _apply_rigid_boundaries(self, pressure: np.ndarray,
                              velocity: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Apply rigid (reflective) boundary conditions."""
        vx, vy, vz = velocity
        
        # Set normal velocity to zero at boundaries
        vx[0, :, :] = 0
        vx[-1, :, :] = 0
        vy[:, 0, :] = 0
        vy[:, -1, :] = 0
        vz[:, :, 0] = 0
        vz[:, :, -1] = 0
        
        # Zero-gradient for pressure at boundaries
        pressure[0, :, :] = pressure[1, :, :]
        pressure[-1, :, :] = pressure[-2, :, :]
        pressure[:, 0, :] = pressure[:, 1, :]
        pressure[:, -1, :] = pressure[:, -2, :]
        pressure[:, :, 0] = pressure[:, :, 1]
        pressure[:, :, -1] = pressure[:, :, -2]
    
    def _apply_periodic_boundaries(self, pressure: np.ndarray,
                                 velocity: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Apply periodic boundary conditions."""
        vx, vy, vz = velocity
        
        # Copy boundaries for periodic conditions
        pressure[0, :, :] = pressure[-2, :, :]
        pressure[-1, :, :] = pressure[1, :, :]
        pressure[:, 0, :] = pressure[:, -2, :]
        pressure[:, -1,, :] = pressure[:, 1, :]
        pressure[:, :, 0] = pressure[:, :, -2]
        pressure[:, :, -1] = pressure[:, :, 1]
        
        vx[0, :, :] = vx[-2, :, :]
        vx[-1, :, :] = vx[1, :, :]
        vy[:, 0, :] = vy[:, -2, :]
        vy[:, -1, :] = vy[:, 1, :]
        vz[:, :, 0] = vz[:, :, -2]
        vz[:, :, -1] = vz[:, :, 1]
    
    def set_boundary_strength(self, strength: float):
        """Set the absorption strength for boundary conditions."""
        self.boundary_strength = max(0.0, min(1.0, strength))
    
    def set_boundary_type(self, boundary_type: str):
        """Set the type of boundary conditions."""
        valid_types = ["absorbing", "pml", "rigid", "periodic"]
        if boundary_type in valid_types:
            self.boundary_type = boundary_type
        else:
            raise ValueError(f"Boundary type must be one of {valid_types}")
