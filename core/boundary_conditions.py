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

# core/boundary_conditions.py (fixed)
import numpy as np
import numba as nb
from numba import prange

class BoundaryConditions:
    """PML and BEM boundary conditions"""
    
    @staticmethod
    @nb.njit(parallel=True)
    def apply_pml_3d(field, pml_thickness, sigma_max=1.0):
        """Apply Perfectly Matched Layer boundary conditions"""
        nx, ny, nz = field.shape[1:]
        
        for i in prange(nx):
            for j in range(ny):
                for k in range(nz):
                    # Calculate distance to boundaries
                    dist_x = min(i, nx - 1 - i)
                    dist_y = min(j, ny - 1 - j)
                    dist_z = min(k, nz - 1 - k)
                    
                    if (dist_x < pml_thickness or 
                        dist_y < pml_thickness or 
                        dist_z < pml_thickness):
                        
                        # Calculate sigma profile
                        sigma_x = sigma_max * ((pml_thickness - min(dist_x, pml_thickness)) / pml_thickness) ** 2
                        sigma_y = sigma_max * ((pml_thickness - min(dist_y, pml_thickness)) / pml_thickness) ** 2
                        sigma_z = sigma_max * ((pml_thickness - min(dist_z, pml_thickness)) / pml_thickness) ** 2
                        
                        sigma = sigma_x + sigma_y + sigma_z
                        
                        # Apply damping
                        for t in range(field.shape[0]):
                            field[t, i, j, k] *= np.exp(-sigma)
    
    @staticmethod
    @nb.njit
    def apply_bem_boundary(pressure, velocity, impedance, normal):
        """Apply Boundary Element Method boundary conditions"""
        # For rigid boundaries: normal velocity = 0
        # For impedance boundaries: p = Z * v_n
        v_normal = (velocity[0] * normal[0] + 
                   velocity[1] * normal[1] + 
                   velocity[2] * normal[2])
        
        if impedance > 1e10:  # Rigid boundary
            velocity[0] -= v_normal * normal[0]
            velocity[1] -= v_normal * normal[1]
            velocity[2] -= v_normal * normal[2]
        else:  # Impedance boundary
            pressure_target = impedance * v_normal
            pressure[0] = pressure_target
