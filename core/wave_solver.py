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

# core/wave_solver.py (fixed)
import numpy as np
import numba as nb
from numba import prange
import dask.array as da
from typing import Tuple, Dict, Any

class WaveSolver3D:
    """3D FDTD wave solver with BEM boundary conditions"""
    
    def __init__(self, config):
        self.config = config
        self.dt = 1.0 / config.sample_rate
        self.dx = config.voxel_size
        self.c = config.speed_of_sound
        
        # CFL condition check
        dt_max = config.voxel_size / (config.speed_of_sound * np.sqrt(3))
        if self.dt > config.cfl_number * dt_max:
            raise ValueError("CFL condition violated")
    
    @staticmethod
#    @nb.njit(parallel=True, fastmath=True)
    def update_pressure_3d(pressure, velocity_x, velocity_y, velocity_z, 
                          dt, dx, c, rho, impedance_map, material_map):
        """Update pressure field using FDTD"""
        nx, ny, nz = pressure.shape[1:]
        
        for i in prange(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    # Pressure update from velocity divergence
                    div_v = ((velocity_x[0, i+1, j, k] - velocity_x[0, i-1, j, k]) +
                            (velocity_y[0, i, j+1, k] - velocity_y[0, i, j-1, k]) +
                            (velocity_z[0, i, j, k+1] - velocity_z[0, i, j, k-1])) / (2 * dx)
                    
                    material_id = material_map[i, j, k]
                    impedance = impedance_map[material_id]
                    
                    pressure[1, i, j, k] = (pressure[0, i, j, k] - 
                                           dt * impedance * c * div_v)
    
    @staticmethod
#    @nb.njit(parallel=True, fastmath=True)
    def update_velocity_3d(velocity_x, velocity_y, velocity_z, pressure,
                          dt, dx, rho):
        """Update velocity fields"""
        nx, ny, nz = pressure.shape[1:]
        
        for i in prange(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    # Velocity updates from pressure gradient
                    dp_dx = (pressure[0, i+1, j, k] - pressure[0, i-1, j, k]) / (2 * dx)
                    dp_dy = (pressure[0, i, j+1, k] - pressure[0, i, j-1, k]) / (2 * dx)
                    dp_dz = (pressure[0, i, j, k+1] - pressure[0, i, j, k-1]) / (2 * dx)
                    
                    velocity_x[1, i, j, k] = (velocity_x[0, i, j, k] - 
                                             dt * dp_dx / rho)
                    velocity_y[1, i, j, k] = (velocity_y[0, i, j, k] - 
                                             dt * dp_dy / rho)
                    velocity_z[1, i, j, k] = (velocity_z[0, i, j, k] - 
                                             dt * dp_dz / rho)

    @staticmethod
#    @nb.njit(parallel=True)
    def apply_source(pressure, source_pos, source_value, radius=2):
        """Apply source excitation to pressure field"""
        x, y, z = source_pos
        for i in prange(max(1, x-radius), min(pressure.shape[1]-1, x+radius+1)):
            for j in range(max(1, y-radius), min(pressure.shape[2]-1, y+radius+1)):
                for k in range(max(1, z-radius), min(pressure.shape[3]-1, z+radius+1)):
                    dist = np.sqrt((i-x)**2 + (j-y)**2 + (k-z)**2)
                    if dist <= radius:
                        weight = 1.0 - (dist / radius)
                        pressure[1, i, j, k] += source_value * weight
                        print('WaveSolver3D.apply_source :', pressure[1, i, j, k])
