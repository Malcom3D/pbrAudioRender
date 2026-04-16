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
import threading
from dask import delayed

from ..lib.ray_data import SIMDRay

class WavePropagator:
    """Optimized wave propagator using SIMD and parallel processing"""
    
    def __init__(self, entity_manager, combo):
        self.entity_manager = entity_manager
        self.source_idx, self.output_idx = combo
        self.config = entity_manager.get('config')
        
        # Precompute frequency bands
        self.freq_bands = entity_manager.get('frequency_bands').get_bands()
        self.num_bands = len(self.freq_bands)
        
        # Initialize ray pools for reuse (object pooling)
        self.ray_pool = []
        self.max_rays = self.config.system.number_of_rays * 100  # Buffer
        
        # Thread-local storage for parallel processing
        self.local_data = threading.local()

    def _prepare_scene(self, frame_idx: int):
        pass

    @delayed
    def compute_frame(self, frame_idx):
        """Compute impulse response for a single frame"""
        # Get scene data for this frame
        scene_data = self._prepare_scene(frame_idx)
        
        # Generate initial rays
        n_rays = self.config.system.number_of_rays
        rays = self._generate_initial_rays(frame_idx, n_rays)
        
        # Main propagation loop
        max_interactions = self.config.wave_propagation.max_interactions
        impulse_response = np.zeros((self.num_bands, 1024))  # Time bins
        
        for interaction in range(max_interactions):
            if not rays:
                break
            
            # Batch process all rays
            origins = np.array([r.origin for r in rays])
            directions = np.array([r.direction for r in rays])
            
            # Intersect with scene
            distances, points, normals, obj_ids = intersect_rays_batch(origins, directions, scene_data['triangles'], scene_data['triangle_normals'], scene_data['triangle_obj_ids'])
            
            # Process interactions
            new_rays = []
            for i, ray in enumerate(rays):
                if not obj_ids[i] in [-1, -2]:  # Hit
                    # Apply boundary interactions
                    reflected, refracted = self._process_interaction(ray, points[i], normals[i], obj_ids[i])
                    
                    if reflected is not None:
                        new_rays.append(reflected)
                    if refracted is not None:
                        new_rays.append(refracted)
                    
                    # Add to impulse response if hit output
                    if obj_ids[i] == -3:  # Output object
                        self._accumulate_to_ir(ray, distances[i], impulse_response)
                else:
                    # Ray hit acoustc domain or source
                    pass
            
            rays = new_rays
        
        return impulse_response
    
    def _generate_initial_rays(self, frame_idx, n_rays):
        """Generate initial rays with importance sampling"""
        source_pos = self._get_source_position(frame_idx)
        output_pos = self._get_output_position(frame_idx)
        
        # Importance sampling: more rays towards output
        rays = []
        for i in range(n_rays):
            # Blend between isotropic and directed sampling
            if np.random.random() < 0.3:  # 30% directed rays
                direction = output_pos - source_pos
                direction = direction / np.linalg.norm(direction)
                # Add some jitter
                direction += np.random.normal(0, 0.1, 3)
                direction = direction / np.linalg.norm(direction)
            else:
                # Isotropic direction
                direction = self._random_isotropic_direction()
            
            ray = SIMDRay(source_pos.copy(), direction, self.num_bands)
            rays.append(ray)
        
        return rays
    
    @staticmethod
    @nb.njit(fastmath=True)
    def _random_isotropic_direction():
        """Fast isotropic direction generation"""
        # Marsaglia method
        while True:
            x1 = np.random.uniform(-1, 1)
            x2 = np.random.uniform(-1, 1)
            s = x1*x1 + x2*x2
            if s < 1:
                break
        
        x = 2 * x1 * np.sqrt(1 - s)
        y = 2 * x2 * np.sqrt(1 - - s)
        z = 1 - 2 * s
        
        return np.array([x, y, z])
    
    def _process_interaction(self, ray, point, normal, obj_id):
        """Process boundary interaction with SIMD operations"""
        # Get material properties
        material = self._get_material(obj_id)
        
        # Compute reflection (vectorized for all frequency bands)
        reflected_dir = ray.direction - 2 * np.dot(ray.direction, normal) * normal
        
        # Create reflected ray
        reflected = SIMDRay(point, reflected_dir, self.num_bands)
        reflected.energy = ray.energy.copy()
        reflected.phase = ray.phase.copy()
        reflected.reflection_count = ray.reflection_count + 1
        
        # Apply frequency-dependent reflection coefficients
        for band_idx in range(self.num_bands):
            low_freq, high_freq = self.freq_bands[band_idx]
            coeff = material.get_reflection_coeff(low_freq, high_freq)
            reflected.energy[band_idx] *= coeff
        
        # Compute refraction if applicable
        refracted = None
        if material.transparent:
            # Snell's law
            n1 = self._get_medium_index(ray.medium_idx)
            n2 = material.refractive_index
            
            cos_theta = -np.dot(ray.direction, normal)
)
            sin_theta2 = (n1/n2)**2 * (1 - cos_theta*cos_theta)
            
            if sin_theta2 <= 1:  # Not total internal reflection
                cos_theta2 = np.sqrt(1 - sin_theta2)
                refracted_dir = (n1/n2) * ray.direction + ((n1/n2) * cos_theta - cos_theta2) * normal
                refracted = SIMDRay(point, refracted_dir, self.num_bands)
                refracted.energy = ray.energy.copy()
                refracted.phase = ray.phase.copy()
                refracted.medium_idx = obj_id  # Enter new medium
        
        return reflected, refracted
    
    @nb.njit(parallel=True)
    def _accumulate_to_ir(self, ray, distance, impulse_response):
        """Accumulate ray contribution to impulse response (SIMD)"""
        speed_of_sound = 343.0  # Get from config
        time_bin = int(distance / speed_of_sound * self.config.system.sample_rate)
        
        if 0 <= time_bin < impulse_response.shape[1]:
            for band_idx in prange(self.num_bands):
                impulse_response[band_idx, time_bin] += ray.energy[band_idx]
