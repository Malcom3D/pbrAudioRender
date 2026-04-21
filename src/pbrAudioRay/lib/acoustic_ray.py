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
        self.path_length = np.zeros(n_rays, dtype=np.float32)
        
        # Frequency-dependent properties
        self.energy = np.ones((n_rays, n_freq_bands), dtype=np.complex64)
        self.phase = np.zeros((n_rays, n_freq_bands), dtype=np.float32)
        
        # Material interaction history
        self.interaction_count = np.zeros(n_rays, dtype=np.int32)
        self.interaction_types = np.zeros((n_rays, max_depth), dtype=np.int32)
        self.interaction_coeffs = np.zeros((n_rays, max_depth, n_freq_bands, 5), dtype=np.float32)
        self.interaction_normals = np.zeros((n_rays, max_depth, 3), dtype=np.float32)
        self.interaction_points = np.zeros((n_rays, max_depth, 3), dtype=np.float32)
        
        # Output hits storage
        self.output_hits = {
            'positions': [],
            'energies': [],
            'phases': [],
            'path_lengths': [],
            'interaction_counts': []
        }
        
        # Gradient information
        self.gradients = np.zeros((n_rays, n_freq_bands, 4), dtype=np.float32)
        
        # Intersection results
        self.hit = np.zeros(n_rays, dtype=np.bool_)
        self.distance = np.full(n_rays, np.inf, dtype=np.float32)
        self.object_idx = np.full(n_rays, -1, dtype=np.int32)
        self.face_idx = np.full(n_rays, -1, dtype=np.int32)
        self.barycentric = np.zeros((n_rays, 3), dtype=np.float32)
        self.normal = np.zeros((n_rays, 3), dtype=np.float32)
        self.point = np.zeros((n_rays, 3), dtype=np.float32)
    
    def store_output_hit(self, ray_idx: int, hit_point: np.ndarray):
        """
        Store output hit information for impulse response.
        
        Args:
            ray_idx: Index of ray that hit output
            hit_point: Point where ray hit output
        """
        self.output_hits['positions'].append(hit_point)
        self.output_hits['energies'].append(self.energy[ray_idx].copy())
        self.output_hits['phases'].append(self.phase[ray_idx].copy())
        self.output_hits['path_lengths'].append(self.path_length[ray_idx])
        self.output_hits['interaction_counts'].append(self.interaction_count[ray_idx])
        
        # Deactivate ray
        self.active[ray_idx] = False
    
    def get_impulse_response(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute impulse response from stored output hits.
        
        Returns:
            Tuple of (time_delays, frequency_responses)
        """
        n_hits = len(self.output_hits['positions'])
        if n_hits == 0:
            return np.array([]), np.array([])
        
        # Convert path lengths to time delays
        sound_speed = 343.0  # Default, should be configurable
        time_delays = np.array(self.output_hits['path_lengths']) / sound_speed
        
        # Sum energies for each frequency band
        energies = np.array(self.output_hits['energies'])
        frequency_responses = np.sum(np.abs(energies), axis=0)
        
        return time_delays, frequency_responses
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def update_rays_batch(origins: np.ndarray, directions: np.ndarray,
                         energies: np.ndarray, phases: np.ndarray,
                         distances: np.ndarray, frequencies: np.ndarray,
                         sound_speed: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch update ray properties after propagation.
        
        Args:
            origins: Ray origins
            directions: Ray directions
            energies: Ray energies
            phases: Ray phases
            distances: Propagation distances
            frequencies: Frequency bands
            sound_speed: Speed of sound
            
        Returns:
            Tuple of (updated_energies, updated_phases)
        """
        n_rays, n_bands = energies.shape
        updated_energies = energies.copy()
        updated_phases = phases.copy()
        
        for i in nb.prange(n_rays):
            if distances[i] > 0:
                # Apply inverse square law
                attenuation = 1.0 / (distances[i] * distances[i])
                updated_energies[i] *= attenuation
                
                # Apply phase shift
                for j in range(n_bands):
                    phase_shift = 2.0 * np.pi * frequencies[j] * distances[i] / sound_speed
                    updated_phases[i, j] = np.mod(phases[i, j] + phase_shift + np.pi, 2.0 * np.pi) - np.pi
        
        return updated_energies, updated_phases
