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

"""
Boundary conditions for the acoustic simulation domain.
Implements various boundary conditions including PML, ABC, and open boundaries.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import numba as nb

from ..utils.parallel_proc import configure_numba
from ..utils.gpu_acceleration import GPUManager


class BoundaryConditions:
    """Manages boundary conditions for the simulation domain"""
    
    def __init__(self, config, gpu_manager: Optional[GPUManager] = None):
        self.config = config
        self.gpu = gpu_manager
        self.jit = configure_numba(parallel=True)
        
        # Boundary condition parameters
        self.pml_thickness = 10  # Number of PML layers
        self.abc_coefficient = 0.1  # Absorbinging boundary coefficient
        self.boundary_type = "pml"  # pml, abc, open, rigid
        
        # PML parameters
        self.pml_sigma_max = 2.0
        self.pml_alpha = 0.0
        self.pml_kappa_max = 1.0
        
        # Initialize PML arrays if needed
        self.pml_sigma_x = None
        self.pml_sigma_y = None
        self.pml_sigma_z = None
        
    def update_step(self, fdtd_fields: Dict[str, np.ndarray],
                   damped_fields: Dict[str, np.ndarray],
                   soxel_grid) -> Dict[str, np.ndarray]:
        """
        Apply boundary conditions to the acoustic fields.
        
        Args:
            fdtd_fields: Fields from FDTD solver
            damped_fields: Fields after damping
            soxel_grid: Current SoxelGrid state
        
        Returns:
            Fields with boundary conditions applied
        """
        result_fields = damped_fields.copy()
        
        if self.boundary_type == "pml":
            result_fields = self._apply_pml_boundary(result_fields, soxel_grid)
        elif self.boundary_type == "abc":
            result_fields = self._apply_abc_boundary(result_fields, soxel_grid)
        elif self.boundary_type == "open":
            result_fields = self._apply_open_boundary(result_fields, soxel_grid)
        elif self.boundary_type == "rigid":
            result_fields = self._apply_rigid_boundary(result_fields, soxel_grid)
        
        return result_fields
    
    def _apply_pml_boundary(self, fields: Dict[str, np.ndarray],
                          soxel_grid) -> Dict[str, np.ndarray]:
        """Apply Perfectly Matched Layer boundary conditions"""
        if self.pml_sigma_x is None:
            self._initialize_pml_arrays(soxel_grid.shape)
        
        result_fields = fields.copy()
        
        if self.gpu and self.gpu.config.use_gpu:
            result_fields = self._apply_pml_gpu(result_fields, soxel_grid)
        else:
            result_fields = self._apply_pml_cpu(result_fields, soxel_grid)
        
        return result_fields
    
    def _initialize_pml_arrays(self, shape: Tuple[int, int, int]):
        """Initialize PML absorption arrays"""
        nx, ny, nz = shape
        pml_thickness = self.pml_thickness
        
        # X-direction PML
        self.pml_sigma_x = np.zeros((nx, ny, nz), dtype=np.float32)
        for i in range(pml_thickness):
            # Left boundary
            sigma = self.pml_sigma_max * ((pml_thickness - i) / pml_thickness) ** 2
            self.pml_sigma_x[i, :, :] = sigma
            
            # Right boundary
            self.pml_sigma_x[nx - 1 - i, :, :] = sigma
        
        # Y-direction PML
        self.pml_sigma_y = np.zeros((nx, ny, nz), dtype=np.float32)
        for j in range(pml_thickness):
            # Front boundary
            sigma = self.pml_sigma_max * ((pml_thickness - j) / pml_thickness) ** 2
            self.pml_sigma_x[:, j, :] = sigma
            
            # Back boundary
            self.pml_sigma_x[:, ny - 1 - j, :] = sigma
        
        # Z-direction PML
        self.pml_sigma_z = np.zeros((nx, ny, nz), dtype=np.float32)
        for k in range(pml_thickness):
            # Bottom boundary
            sigma = self.pml_sigma_max * ((pml_thickness - k) / pml_thickness) ** 2
            self.pml_sigma_z[:, :, k] = sigma
            
            # Top boundary
            self.pml_sigma_z[:, :, nz - 1 - k] = sigma
    
    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def _apply_pml_cpu(fields: Dict[str, np.ndarray],
                      soxel_grid, pml_sigma_x, pml_sigma_y, pml_sigma_z,
                      dt: float, dx: float) -> Dict[str, np.ndarray]:
        """CPU implementation of PML boundary conditions"""
        pressure = fields['pressure'].copy()
        vx = fields['velocity_x'].copy()
        vy = fields['velocity_y'].copy()
        vz = fields['velocity_z'].copy()
        
        shape = pressure.shape
        nx, ny, nz = shape
        
        for i in nb.prange(nx):
            for j in range(ny):
                for k in range(nz):
                    # Apply PML absorption to pressure
                    if pml_sigma_x[i, j, k] > 0:
                        # Calculate pressure gradient
                        if i > 0 and i < nx - 1:
                            dp_dx = (pressure[i+1, j, k] - pressure[i-1, j, k]) / (2 * dx)
                            pressure[i, j, k] -= dt * pml_sigma_x[i, j, k] * dp_dx
                    
                    if pml_sigma_y[i, j, k] > 0:
                        if j > 0 and j < ny - 1:
                            dp_dy = (pressure[i, j+1, k] - pressure[i, j-1, k]) / (2 * dx)
                            pressure[i, j, k] -= dt * pml_sigma_y[i, j, k] * dp_dy
                    
                    if pml_sigma_z[i, j, k] > 0:
                        if k > 0 and k < nz - 1:
                            dp_dz = (pressure[i, j, k+1] - pressure[i, j, k-1]) / (2 * dx)
                            pressure[i, j, k] -= dt * pml_sigma_z[i, j, k] * dp_dz
        
        return {
            'pressure': pressure,
            'velocity_x': vx,
            'velocity_y': vy,
            'velocity_z': vz
        }
    
    def _apply_abc_boundary(self, fields: Dict[str, np.ndarray],
                          soxel_grid) -> Dict[str, np.ndarray]:
        """Apply Absorbing Boundary Conditions"""
        result_fields = fields.copy()
        shape = soxel_grid.shape
        nx, ny, nz = shape
        
        # Apply ABC to each boundary face
        abc_coeff = self.abc_coefficient
        
        # X boundaries
        for i in range(2):  # Left and right boundaries
            for j in range(ny):
                for k in range(nz):
                    if i == 0:  # Left boundary
                        result_fields['pressure'][i, j, k] *= (1 - abc_coeff)
                    else:  # Right boundary
                        result_fields['pressure'][nx-1-i, j, k] *= (1 - abc_coeff)
        
        # Y boundaries
        for j in range(2):  # Front and back boundaries
            for i in range(nx):
                for k in range(nz):
                    if j == 0:  # Front boundary
                        result_fields['pressure'][i, j, k] *= (1 - abc_coeff)
                    else:  # Back boundary
                        result_fields['pressure'][i, ny-1-j, k] *= (1 - abc_coeff)
        
        # Z boundaries
        for k in range(2):  # Bottom and top boundaries
            for i in range(nx):
                for j in range(ny):
                    if k == 0:  # Bottom boundary
                        result_fields['pressure'][i, j, k] *= (1 - abc_coeff)
                    else:  # Top boundary
                        result_fields['pressure'][i, j, nz-1-k] *= (1 - abc_coeff)
        
        return result_fields
    
    def _apply_open_boundary(self, fields: Dict[str, np.ndarray],
                           soxel_grid) -> Dict[str, np.ndarray]:
        """Apply open boundary conditions (zero gradient)"""
        result_fields = fields.copy()
        shape = soxel_grid.shape
        nx, ny, nz = shape
        
        # Set boundary values to adjacent interior values
        # X boundaries
        result_fields['pressure'][0, :, :] = result_fields['pressure'][1, :, :]
        result_fields['pressure'][nx-1, :, :] = result_fields['pressure'][nx-2, :, :]
        
        # Y boundaries
        result_fields['pressure'][:, 0, :] = result_fields['pressure'][:, 1, :]
        result_fields['pressure'][:, ny-1, :] = result_fields['pressure'][:, ny-2, :]
        
        # Z boundaries
        result_fields['pressure'][:, :, 0] = result_fields['pressure'][:, :, 1]
        result_fields['pressure'][:, :, nz-1] = result_fields['pressure'][:, :, nz-2]
        
        return result_fields
    
    def _apply_rigid_boundary(self, fields: Dict[str, np.ndarray],
                            soxel_grid) -> Dict[str, np.ndarray]:
        """Apply rigid boundary conditions (zero velocity)"""
        result_fields = fields.copy()
        shape = soxel_grid.shape
        nx, ny, nz = shape
        
        # Set normal velocity to zero at boundaries
        # X boundaries
        result_fields['velocity_x'][0, :, :] = 0.0
        result_fields['velocity_x'][nx-1, :, :] = 0.0
        
        # Y boundaries
        result_fields['velocity_y'][:, 0, :] = 0.0
        result_fields['velocity_y'][:, ny-1, :] = 0.0
        
        # Z boundaries
        result_fields['velocity_z'][:, :, 0] = 0.0
        result_fields['velocity_z'][:, :, nz-1] = 0.0
        
        return result_fields
    
    def _apply_pml_gpu(self, fields: Dict[str, np.ndarray],
                      soxel_grid) -> Dict[str, np.ndarray]:
        """GPU implementation of PML boundary conditions"""
        # For now, fall back to CPU implementation
        dt = 1.0 / soxel_grid.config.sample_rate
        dx = soxel_grid.voxel_size
        
        return self._apply_pml_cpu(fields, soxel_grid, 
                                 self.pml_sigma_x, self.pml_sigma_y, self.pml_sigma_z,
                                 dt, dx)
    
    def set_boundary_type(self, boundary_type: str):
        """Set the boundary condition type"""
        valid_types = ["pml", "abc", "open", "rigid"]
        if boundary_type in valid_types:
            self.boundary_type = boundary_type
        else:
            raise ValueError(f"Invalid boundary type: {boundary_type}. Must be one of {valid_types}")

