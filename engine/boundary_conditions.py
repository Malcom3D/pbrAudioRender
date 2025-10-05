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
import scipy.signal as signal
from typing import Tuple, Optional
import warnings

class BoundaryConditions:
    """
    Implements a reflectionless and fully absorbing boundary condition
    for 3D acoustic wave propagation using Perfectly Matched Layer (PML) technique.
    """

    def __init__(self, grid_shape, pml_thickness, absorption_coeff, frequency_range, sample_rate, gpu_config):
        """
        Initialize the open boundary conditions.

        Parameters:
        -----------
        grid_shape : Tuple[int, int, int]
            Shape of the 3D simulation grid (nx, ny, nz)
        pml_thickness : int
            Thickness of the Perfectly Matched Layer in grid points
        absorption_coeff : float
            Absorption coefficient (0.0 to 1.0)
        frequency_range : Tuple[float, float]
            Frequency range for optimized absorption (Hz)
        sample_rate : float
            sample_rate of sound in simulation
        gpu_config: None
            GPUConfig class
        """
        self.grid_shape = grid_shape
        self.pml_thickness = pml_thickness
        self.absorption_coeff = np.clip(absorption_coeff, 0.0, 1.0)
        self.frequency_range = frequency_range
        # Time step for simulation
        self.dt = 1/sample_rate

        self.gpu_config = gpu_config

        # Initialize PML absorption profiles
        self._init_pml_profiles()

        # Initialize boundary fields
        self._initialize_boundary_fields()

        print(f"BoundaryConditions initialized: {grid_shape}, PML thickness: {pml_thickness}")

    def _init_pml_profiles(self):
        """Initialize PML absorption profiles for all boundaries."""
        nx, ny, nz = self.grid_shape
        pml_t = self.pml_thickness

        # Create absorption profiles using polynomial function
        self.pml_x = self._create_pml_profile(nx, pml_t)
        self.pml_y = self._create_pml_profile(ny, pml_t)
        self.pml_z = self._create_pml_profile(nz, pml_t)

        # Combine into 3D absorption masks
        self.absorption_mask_x = self._create_3d_mask(self.pml_x, ny, nz, axis=0)
        self.absorption_mask_y = self._create_3d_mask(self.pml_y, nx, nz, axis=1)
        self.absorption_mask_z = self._create_3d_mask(self.pml_z, nx, ny, axis=2)

        # Combined absorption mask
        self.combined_mask = (
            self.absorption_mask_x *
            self.absorption_mask_y *
            self.absorption_mask_z
        )

    def _create_pml_profile(self, grid_size: int, pml_thickness: int) -> np.ndarray:
        """Create 1D PML absorption profile."""
        profile = np.ones(grid_size, dtype=np.float64)

        # Left boundary
        for i in range(pml_thickness):
            # Polynomial absorption profile
            sigma = self.absorption_coeff * ((pml_thickness - i) / pml_thickness) ** 3
            profile[i] = 1.0 - sigma

        # Right boundary
        for i in range(grid_size - pml_thickness, grid_size):
            sigma = self.absorption_coeff * ((i - (grid_size - pml_thickness - 1)) / pml_thickness) ** 3
            profile[i] = 1.0 - sigma

        return profile

    def _create_3d_mask(self, profile: np.ndarray, dim1: int, dim2: int, axis: int) -> np.ndarray:
        """Create 3D absorption mask from 1D profile."""
        if axis == 0:
            mask = np.ones((len(profile), dim1, dim2), dtype=np.float64)
            for i in range(len(profile)):
                mask[i, :, :] = profile[i]
        elif axis == 1:
            mask = np.ones((dim1, len(profile), dim2), dtype=np.float64)
            for i in range(len(profile)):
                mask[:, i, :] = profile[i]
        else:  # axis == 2
            mask = np.ones((dim1, dim2, len(profile)), dtype=np.float64)
            for i in range(len(profile)):
                mask[:, :, i] = profile[i]
        return mask

    def _initialize_boundary_fields(self):
        """Initialize fields for boundary condition application."""
        nx, ny, nz = self.grid_shape

        # Previous pressure fields for temporal filtering
        self.prev_pressure = np.zeros((nx, ny, nz), dtype=np.float64)
        self.prev2_pressure = np.zeros((nx, ny, nz), dtype=np.float64)

        # Boundary flux storage
        self.boundary_flux = np.zeros((6, max(nx, ny, nz), max(nx, ny, nz)), dtype=np.float64)

        # Frequency-dependent absorption factors
        self._setup_frequency_absorption()

    def _setup_frequency_absorption(self):
        """Setup frequency-dependent absorption parameters."""
        f_min, f_max = self.frequency_range

        # Calculate optimal absorption parameters based on frequency range
        # This is a simplified model - can be extended with more sophisticated models
        center_freq = np.sqrt(f_min * f_max)
        wavelength = 343.0 / center_freq  # assuming speed of sound = 343 m/s

        # Adjust absorption based on frequency content
        self.freq_absorption_factor = np.clip(wavelength / (self.pml_thickness * 0.1), 0.1, 1.0)

#    @jit(nopython=True, parallel=True, fastmath=True)
    def apply(self, pressure_field: np.ndarray, velocity_fields: np.ndarray) -> np.ndarray:
        """
        Apply open boundary conditions to the pressure field.

        Parameters:
        -----------
        pressure_field : np.ndarray
            Current pressure field (3D array)
        velocity_fields : Tuple
            Tuple of velocity field components (vx, vy, vz)

        Returns:
        --------
        np.ndarray: Pressure field with boundary conditions applied
        """
        nx, ny, nz = pressure_field.shape
        pml_t = self.pml_thickness

        # Create a copy to work with
        result = pressure_field.copy()

        # Apply PML absorption
        self._apply_pml_absorption(result, pml_t)

        # Apply outgoing wave condition at boundaries
        self._apply_outgoing_wave_condition(result, pressure_field, pml_t)

        return result

#    @jit(nopython=True, parallel=True)
    def _apply_pml_absorption(self, field: np.ndarray, pml_thickness: int):
        """Apply PML absorption to the field boundaries."""
        nx, ny, nz = field.shape

        # Apply absorption in all boundary regions
        for i in prange(nx):
            for j in prange(ny):
                for k in prange(nz):
                    # Calculate distance to nearest boundary
                    dist_x = min(i, nx - 1 - i)
                    dist_y = min(j, ny - 1 - j)
                    dist_z = min(k, nz - 1 - k)

                    # Check if in PML region
                    if (dist_x < pml_thickness or
                        dist_y < pml_thickness or
                        dist_z < pml_thickness):

                        # Calculate absorption factor based on distance
                        absorption = 1.0
                        if dist_x < pml_thickness:
                            absorption *= 1.0 - self.absorption_coeff * ((pml_thickness - dist_x) / pml_thickness) ** 2
                        if dist_y < pml_thickness:
                            absorption *= 1.0 - self.absorption_coeff * ((pml_thickness - dist_y) / pml_thickness) ** 2
                        if dist_z < pml_thickness:
                            absorption *= 1.0 - self.absorption_coeff * ((pml_thickness - dist_z) / pml_thickness) ** 2

                        field[i, j, k] *= max(absorption, 0.01)  # Prevent complete zero

#    @jit(nopython=True)
    def _apply_outgoing_wave_condition(self, result: np.ndarray,
                                     prev_field: np.ndarray,
                                     pml_thickness: int):
        """Apply outgoing wave condition using one-way wave equation approximation."""
        nx, ny, nz = result.shape

        # Simple first-order absorbing boundary condition
        # This can be extended to higher-order methods
        c = 343.0  # Speed of sound

        # Apply to each face
        for i in range(pml_thickness):
            # Left boundary (x=0)
            if i < nx:
                result[i, :, :] -= (c * self.dt) * (prev_field[i+1, :, :] - prev_field[i, :, :])

            # Right boundary (x=max)
            if nx-1-i >= 0:
                result[nx-1-i, :, :] -= (c * self.dt) * (prev_field[nx-1-i, :, :] - prev_field[nx-2-i, :, :])

        for j in range(pml_thickness):
            # Front boundary (y=0)
            if j < ny:
                result[:, j, :] -= (c * self.dt) * (prev_field[:, j+1, :] - prev_field[:, j, :])

            # Back boundary (y=max)
            if ny-1-j >= 0:
                result[:, ny-1-j, :] -= (c * self.dt) * (prev_field[:, ny-1-j, :] - prev_field[:, ny-2-j, :])

        for k in range(pml_thickness):
            # Bottom boundary (z=0)
            if k < nz:
                result[:, :, k] -= (c * self.dt) * (prev_field[:, :, k+1] - prev_field[:, :, k])

            # Top boundary (z=max)
            if nz-1-k >= 0:
                result[:, :, nz-1-k] -= (c * self.dt) * (prev_field[:, :, nz-1-k] - prev_field[:, :, nz-2-k])

    def update(self, pressure_field: np.ndarray):
        """Update boundary state for temporal filtering."""
        self.prev2_pressure = self.prev_pressure.copy()
        self.prev_pressure = pressure_field.copy()

    def calculate_boundary_energy_loss(self, pressure_field: np.ndarray) -> float:
        """
        Calculate the amount of energy lost through boundaries.

        Parameters:
        -----------
        pressure_field : np.ndarray
            Current pressure field

        Returns:
        --------
        float: Energy loss factor (0.0 to 1.0)
        """
        total_energy = np.sum(pressure_field ** 2)
        boundary_energy = self._calculate_boundary_energy(pressure_field)

        if total_energy > 0:
            return boundary_energy / total_energy
        return 0.0

    @jit(nopython=True)
    def _calculate_boundary_energy(self, field: np.ndarray) -> float:
        """Calculate energy in boundary regions."""
        nx, ny, nz = field.shape
        pml_t = self.pml_thickness
        boundary_energy = 0.0

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if (i < pml_t or i >= nx - pml_t or
                        j < pml_t or j >= ny - pml_t or
                        k < pml_t or k >= nz - pml_t):
                        boundary_energy += field[i, j, k] ** 2

        return boundary_energy

    def get_boundary_efficiency(self) -> dict:
        """
        Get boundary absorption efficiency metrics.

        Returns:
        --------
        dict: Dictionary containing efficiency metrics
        """
        return {
            'pml_thickness': self.pml_thickness,
            'absorption_coefficient': self.absorption_coeff,
            'frequency_optimization_factor': self.freq_absorption_factor,
            'boundary_coverage': self._calculate_boundary_coverage()
        }

    def _calculate_boundary_coverage(self) -> float:
        """Calculate what percentage of the grid is covered by boundary conditions."""
        nx, ny, nz = self.grid_shape
        total_points = nx * ny * nz

        boundary_points = (
            2 * self.pml_thickness * ny * nz +  # x-boundaries
            2 * self.pml_thickness * nx * nz +  # y-boundaries
            2 * self.pml_thickness * nx * ny    # z-bound-boundaries
        )

        # Subtract overlapping corner regions (approximate)
        corner_overlap = 8 * (self.pml_thickness ** 3)
        boundary_points -= corner_overlap

        return boundary_points / total_points
