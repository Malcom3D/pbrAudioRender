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
import numba as nb
from numba import prange
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class AmbisonicEncoder:
    """
    Efficient ambisonic encoder for ray tracing output.
    Optimized for CPU with SIMD operations.
    """
    order: int = 1
    
    def __post_init__(self):
        self.n_channels = (self.order + 1) ** 2
        
        # Precompute SH basis functions for common angles
        self._precompute_sh_basis()
    
    def _precompute_sh_basis(self):
        """Precompute spherical harmonic basis functions."""
        # This would store precomputed values for a grid of angles
        # to speed up real-time computation
        pass
    
    @staticmethod
    @nb.jit(nopython=True, fastmath=True, parallel=True)
    def encode_rays(energies: np.ndarray, directions: np.ndarray, order: int) -> np.ndarray:
        """
        Encode ray data to ambisonic channels.
        
        Args:
            energies: Ray energies (n_rays, n_bands)
            directions: Ray directions (n_rays, 3)
            order: Ambisonic order
            
        Returns:
            SH coefficients (n_channels, n_bands)
        """
        n_rays = energies.shape[0]
        n_bands = energies.shape[1] if energies.ndim > 1 else 1
        n_channels = (order + 1) ** 2
        
        # Initialize SH coefficients
        sh_coeffs = np.zeros((n_channels, n_bands), dtype=np.float32)
        
        # Precompute normalization factors
        norm_factors = np.zeros(n_channels, dtype=np.float32)
        norm_factors[0] = 0.5  # W channel
        
        if order >= 1:
            norm_factors[1] = 1.0  # X
            norm_factors[2] = 1.0  # Y
            norm_factors[3] = 1.0  # Z
        
        if order >= 2:
            norm_factors[4] = 0.5  # R
            norm_factors[5] = 0.5  # S
            norm_factors[6] = 1.0  # T
            norm_factors[7] = 1.0  # U
            norm_factors[8] = 0.5  # V
        
        # Process all rays in parallel
        for i in nb.prange(n_rays):
            x, y, z = directions[i, 0], directions[i, 1], directions[i, 2]
            
            # Normalize direction
            norm = np.sqrt(x*x + y*y + z*z)
            if norm > 1e-10:
                x /= norm
                y /= norm
                z /= norm
            
            # Calculate azimuth and elevation
            azimuth = np.arctan2(y, x)
            elevation = np.arcsin(z)
            
            # Precompute trigonometric values
            cos_az = np.cos(azimuth)
            sin_az = np.sin(azimuth)
            cos_el = np.cos(elevation)
            sin_el = np.sin(elevation)
            
            for b in range(n_bands):
                energy = energies[i, b]
                
                # Order 0
                sh_coeffs[0, b] += energy * norm_factors[0]
                
                if order >= 1:
                    # Order 1
                    sh_coeffs[1, b] += energy * cos_az * cos_el * norm_factors[1]
                    sh_coeffs[2, b] += energy * sin_az * cos_el * norm_factors[2]
                    sh_coeffs[3, b] += energy * sin_el * norm_factors[3]
                
                if order >= 2:
                    # Order 2
                    cos2_az = np.cos(2 * azimuth)
                    sin2_az = np.sin(2 * azimuth)
                    cos2_el = np.cos(2 * elevation)
                    
                    sh_coeffs[4, b] += energy * cos2_az * cos_el * cos_el * norm_factors[4]
                    sh_coeffs[5, b] += energy * sin2_az * cos_el * cos_el * norm_factors[5]
                    sh_coeffs[6, b] += energy * cos_az * sin_el * cos_el * norm_factors[6]
                    sh_coeffs[7, b] += energy * sin_az * sin_el * cos_el * norm_factors[7]
                    sh_coeffs[8, b] += energy * (3 * sin_el * sin_el - 1) / 2 * norm_factors[8]
        
        return sh_coeffs
    
    @staticmethod
    @nb.jit(nopython=True, fastmath=True)
    def encode_directivity(energies: np.ndarray, directions: np.ndarray, directivity_pattern: np.ndarray, order: int) -> np.ndarray:
        """
        Encode rays with microphone directivity pattern.
        
        Args:
            energies: Ray energies (n_rays, n_bands)
            directions: Ray directions (n_rays, 3)
            directivity_pattern: Precomputed directivity weights (n_rays,)
            order: Ambisonic order
            
        Returns:
            SH coefficients (n_channels, n_bands)
        """
        # Apply directivity weighting
        return AmbisonicEncoder.encode_rays(weighted_energies, directions, order)
