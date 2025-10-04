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
from typing import Tuple

class FDTDSolver:
    def __init__(self, grid_shape, dx, sample_rate, c, gpu_config):
        self.grid_shape = grid_shape
        self.dx = dx
        self.dt = 1.0 / sample_rate
        self.c = c
        self.CFL = c * self.dt / dx
        
        # Stability check
        if self.CFL > 1.0/np.sqrt(3):  # 3D stability condition
            raise ValueError(f"CFL condition violated: {self.CFL:.3f} > {1.0/np.sqrt(3):.3f}")
            
        self.gpu_config = gpu_config
        self.parallel_target = gpu_config.get_parallel_target()
        
#    @jit(nopython=True, parallel=True)
    def solve_time_step(self, soxels, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        """Solve one time step using 3D FDTD"""
        nx, ny, nz = self.grid_shape
        
        # Initialize fields
        pressure_new = np.zeros((nx, ny, nz))
        velocity_new = np.zeros((nx, ny, nz, 3))
        
        # Get current pressures and velocities
        pressure_curr = np.zeros((nx, ny, nz))
        velocity_curr = np.zeros((nx, ny, nz, 3))
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    pressure_curr[i,j,k] = soxels[i,j,k].pressure
                    velocity_curr[i,j,k] = soxels[i,j,k].velocity
        
        # FDTD update equations
        for i in prange(1, nx-1):
            for j in prange(1, ny-1):
                for k in prange(1, nz-1):
                    # Update velocity (pressure gradient)
                    velocity_new[i,j,k,0] = velocity_curr[i,j,k,0] - (
                        self.dt / (soxels[i,j,k].physical_props.density * self.dx) * 
                        (pressure_curr[i+1,j,k] - pressure_curr[i-1,j,k])
                    )
                    velocity_new[i,j,k,1] = velocity_curr[i,j,k,1] - (
                        self.dt / (soxels[i,j,k].physical_props.density * self.dx) * 
                        (pressure_curr[i,j+1,k] - pressure_curr[i,j-1,k])
                    )
                    velocity_new[i,j,k,2] = velocity_curr[i,j,k,2] - (
                        self.dt / (soxels[i,j,k].physical_props.density * self.dx) * 
                        (pressure_curr[i,j,k+1] - pressure_curr[i,j,k-1])
                    )
                    
                    # Update pressure (velocity divergence)
                    rho_c2 = soxels[i,j,k].physical_props.density * (
                        soxels[i,j,k].physical_props.speed_of_sound ** 2
                    )
                    
                    pressure_new[i,j,k] = pressure_curr[i,j,k] - (
                        rho_c2 * self.dt / self.dx * (
                            (velocity_curr[i+1,j,k,0] - velocity_curr[i-1,j,k,0]) +
                            (velocity_curr[i,j+1,k,1] - velocity_curr[i,j-1,k,1]) +
                            (velocity_curr[i,j,k+1,2] - velocity_curr[i,j,k-1,2])
                        )
                    )
        
        return pressure_new, velocity_new
