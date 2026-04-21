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
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class AcousticRay:
    """Ray data structure for multiple frequency bands with SIMD optimization"""
    n_rays: int
    n_freq_bands: int
    max_depth: int = 10

    def __post_init__(self):
        """
        Initialize ray data structure for vectorized operations.
        
        Parameters:
        -----------
        n_rays : int
            Number of rays to trace simultaneously
        n_freq_bands : int
            Number of frequency bands (typically 1-8 for acoustic simulation)
        max_depth : int
            Maximum ray recursion depth
        """
        n_rays = self.n_rays
        n_freq_bands = self.n_freq_bands
        max_depth = self.max_depth

        # Ray origin and direction (SIMD-friendly layout)
        self.origins = np.zeros((n_rays, 3), dtype=np.float32)
        self.directions = np.zeros((n_rays, 3), dtype=np.float32)
        
        # Ray state
        self.active = np.ones(n_rays, dtype=np.bool_)
        self.depth = np.zeros(n_rays, dtype=np.int32)
        self.path_length = np np.zeros(n_rays, dtype=np.float32)
        
        # Frequency-dependent properties (complex for phase information)
        self.energy = np.ones((n_rays, n_freq_bands), dtype=np.complex64)
        self.phase = np.zeros((n_rays, n_freq_bands), dtype=np.float32)
        
        # Material interaction history for differentiable tracing
        self.interaction_count = np.zeros(n_rays, dtype=np.int32)
        self.interaction_types = np.zeros((n_rays, max_depth), dtype=np.int32)  # 0: none, 1: reflection, 2: refraction, 3: scattering, 4: diffraction
        self.interaction_coeffs = np.zeros((n_rays, max_depth, n_freq_bands, 5), dtype=np.float32)  # [absorption, reflection, refraction, scattering, diffraction]
        self.interaction_normals = np.zeros((n_rays, max_depth, 3), dtype=np.float32)
        self.interaction_points = np.zeros((n_rays, max_depth, 3), dtype=np.float32)
        
        # Gradient information for differentiable path tracing
        self.gradients = np.zeros((n_rays, n_freq_bands, 4), dtype=np.float32)  # [d/d/dx, d/dy, d/dz, d/dω]
        
        # Intersection results
        self.hit = np.zeros(n_rays, dtype=np.bool_)
        self.distance = np.full(n_rays, np.inf, dtype=np.float32)
        self.object_idx = np.full(n_rays, -1, dtype=np.int32)
        self.face_idx = np.full(n_rays, -1, dtype=np.int32)
        self.barycentric = np.zeros((n_rays, 3), dtype=np.float32)
        self.normal = np.zeros((n_rays, 3), dtype=np.float32)
        self.point = np.zeros((n_rays, 3), dtype=np.float32)

    def reset_ray(self, idx: int):
        """Reset a single ray to initial state"""
        self.active[idx] = True
        self.depth[idx] = 0
        self.path_length[idx] = 0.0
        self.energy[idx] = 1.0 + 0j
        self.phase[idx] = 0.0
        self.interaction_count[idx] = 0
        self.hit[idx] = False
        self.distance[idx] = np.inf
        self.object_idx[idx] = -1

    def add_interaction(self, ray_idx: int, interaction_type: int, coeffs: np.ndarray, 
                       normal: np.ndarray, point: np.ndarray):
        """Record an interaction for differentiable path tracing"""
        if self.interaction_count[ray_idx] < self.max_depth:
            depth = self.interaction_count[ray_idx]
            self.interaction_types[ray_idx, depth] = interaction_type
            self.interaction_coeffs[ray_idx, depth] = coeffs
            self.interaction_normals[ray_idx, depth] = normal
            self.interaction_points[ray_idx, depth] = point
            self.interaction_count[ray_idx] += 1

    @staticmethod
    @nb.njit(nogil=True, fastmath=True, cache=True)
    def compute_path_gradients(origins: np.ndarray, directions: np.ndarray, 
                              interaction_points: np.ndarray, interaction_normals: np.ndarray,
                              interaction_types: np.ndarray, interaction_counts: np.ndarray,
                              frequencies: np.ndarray, sound_speed: float) -> np.ndarray:
        """
        Compute gradients for differentiable path tracing using SIMD.
        
        Based on: https://pub.dega-akustik.de/DAGAAGA_2024/files/upload/paper/489.pdf
        """
        n_rays, n_bands = origins.shape[0], frequencies.shape[0]
        gradients = np.zeros((n_rays, n_bands, 4), dtype=np.float32)
        
        for i in nb.prange(n_rays):
            if interaction_counts[i] == 0:
                continue
                
            # Compute total path length
            total_length = 0.0
            current_point = origins[i]
            
            for j in range(interaction_counts[i]):
                next_point = interaction_points[i, j]
                segment_length = np.sqrt(np.sum((next_point - current_point)**2))
                total_length += segment_length
                current_point = next_point
            
            # Compute gradients for each frequency band
            for k in range(n_bands):
                # Spatial gradient: d/dx, d/dy, d/dz
                # For acoustic path tracing, gradient is related to phase change
                phase_change = 2.0 * np.pi * frequencies[k] * total_length / sound_speed
                
                # Gradient w.r.t. position (simplified - actual depends on interaction types)
                gradients[i, k, 0] = -np.sin(phase_change)  # d/dx
                gradients[i, k, 1] = -np.sin(phase_change)  # d/dy  
                gradients[i, k, 2] = -np.sin(phase_change)  # d/dz
                
                # Frequency gradient: d/dω
                gradients[i, k, 3] = 2.0 * np.pi * total_length / sound_speed * np.cos(phase_change)
        
        return gradients

    @staticmethod
    @nb.njit(nogil=True, fastmath=True, cache=True)
    def apply_material_interaction(energy: np.ndarray, phase: np.ndarray,
                                  interaction_coeffs: np.ndarray, interaction_types: np.ndarray,
                                  frequencies: np.ndarray, distance: float, sound_speed: float):
        """
        Apply material interactions to ray energy and phase using SIMD.
        """
        n_rays, n_bands = energy.shape
        new_energy = energy.copy()
        new_phase = phase.copy()
        
        for i in nb.prange(n_rays):
            for k in range(n_bands):
                # Apply distance-based attenuation (inverse square law)
                if distance[i] > 0:
                    attenuation = 1.0 / (distance[i] * distance[i])
                    new_energy[i, k] *= attenuation
                
                # Apply phase shift due to propagation
                phase_shift = 2.0 * np.pi * frequencies[k] * distance[i] / sound_speed
                new_phase[i, k] = np.mod(phase[i, k] + phase_shift + np.pi, 2.0 * np.pi) - np.pi
                
                # Apply material coefficients from interactions
                for j in range(interaction_coeffs.shape[1]):
                    coeffs = interaction_coeffs[i, j, k]
                    if np.any(coeffs != 0):
                        # Absorption
                        new_energy[i, k] *= (1.0 - coeffs[0])
                        
                        # Reflection/refraction/scattering/diffraction already applied in interaction
                        # These affect direction, not energy directly here
        
        return new_energy, new_phase
