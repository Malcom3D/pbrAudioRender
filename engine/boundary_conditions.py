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
from numba import jit, prange

class BoundaryConditions:
    def __init__(self, grid_shape, pml_thickness, gpu_config):
        self.grid_shape = grid_shape
        self.pml_thickness = pml_thickness
        self.gpu_config = gpu_config
        
        # Initialize PML absorption profiles
        self.sigma_x, self.sigma_y, self.sigma_z = self.init_pml_profiles()
        
    def init_pml_profiles(self):
        """Initialize PML absorption profiles"""
        nx, ny, nz = self.grid_shape
        pml = self.pml_thickness
        
        # Create absorption profiles that increase towards boundaries
        sigma_x = np.zeros(nx)
        sigma_y = np.zeros(ny) 
        sigma_z = np.zeros(nz)
        
        # X-direction PML
        for i in range(pml):
            # Quadratic profile
            sigma_x[i] = ( (pml - i) / pml ) ** 2
            sigma_x[nx-1-i] = ( (pml - i) / pml ) ** 2
            
        # Y-direction PML
        for j in range(pml):
            sigma_y[j] = ( (pml - j) / pml ) ** 2
            sigma_y[ny-1-j] = ( (pml - j) / pml ) ** 2
            
        # Z-direction PML
        for k in range(pml):
            sigma_z[k] = ( (pml - k) / pml ) ** 2
            sigma_z[nz-1-k] = ( (pml - k) / pml ) ** 2
            
        return sigma_x, sigma_y, sigma_z
    
#    @jit(nopython=True, parallel=True)
    def apply(self, pressure: np.ndarray, velocity: np.ndarray) -> tuple:
        """Apply PML boundary conditions"""
        nx, ny, nz = pressure.shape
        
        # Apply PML to velocity field
        for i in prange(nx):
            for j in prange(ny):
                for k in prange(nz):
                    # Apply absorption in X direction
                    if self.sigma_x[i] > 0:
                        velocity[i,j,k,0] *= np.exp(-self.sigma_x[i])
                    
                    # Apply absorption in Y direction  
                    if self.sigma_y[j] > 0:
                        velocity[i,j,k,1] *= np.exp(-self.sigma_y[j])
                    
                    # Apply absorption in Z direction
                    if self.sigma_z[k] > 0:
                        velocity[i,j,k,2] *= np.exp(-self.sigma_z[k])
        
        return pressure, velocity
