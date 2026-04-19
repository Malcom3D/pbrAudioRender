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
from dask import delayed
from typing import List, Tuple
from dataclasses import dataclass

from ..core.entity_manager import EntityManager
from ..lib.embree_scene import EmbreeScene
from ..lib.acoustic_ray import AcousticRay
from ..lib.simd_math import generate_all_directions_batch

@dataclass
class WavePropagator:
    """Optimized wave propagator using SIMD and parallel processing"""
    entity_manager: EntityManager
    combo: Tuple[int, int]
    
    def __post_init__(self):
        self.source_idx, self.output_idx = self.combo
        self.config = self.entity_manager.get('config')
        
        # Get frequency bands
        self.freq_bands = self.entity_manager.get('frequency_bands').get_bands()
        
    @delayed
    def compute(self, frame_idx):
        """Compute impulse response for a single frame"""
        # Get scene data for this frame
        self.embree_scene = EmbreeScene(self.entity_manager, self.combo, frame_idx)
        source_pos = self.embree_scene.src_pos
        output_pos = self.embree_scene.out_pos

        # Generate initial rays data structure
        n_rays = self.config.system.number_of_rays
        n_bands = len(self.freq_bands)
        max_interactions = self.config.wave_propagation.max_interactions
        rays = AcousticRay(n_rays, n_bands, max_interactions)

        # Diffuse source
        for source in self.config.sources:
            if source.idx == self.source_idx:
                if source.type == 'SPERE' and source.size > 0:
                    source_size = source.size
                    n_points = int(np.random.uniform(1, 10, size=1))
                    source_pos = self._source_points(n_points, source_pos, source_size)

#        directions = self._generate_initial_directions(n_rays, source_pos, output_pos)
        
        source_ndim = int(n_rays / source_pos.shape[0])
        source_pos = np.full((source_ndim,3), [source_pos.tolist()], dtype=np.float32)

        n_dirs = source_ndim * source_pos.shape[0]
        directions = self._generate_isotropic_directions(source_pos, output_pos, n_dirs)
        directions = np.array(directions, dtype=np.float32)

        self.first_run(source_pos, directions)

    def first_run(self, source_pos: np.ndarray, directions: np.ndarray):
        print('WavePropagator: first_run')
        scene = self.embree_scene.scene
        scene_info = self.embree_scene.scene_info
        mesh_info = self.embree_scene.mesh_info

        hits = scene.run(source_pos, directions, output=1)

        ray_inter = hits["geomID"] >= 0
        primID = hits["primID"][ray_inter]
        output, hits_obj_idx = self._find_output_and_obj_idx(scene_info[primID])

        hits_coord = (np.vstack(w) * mesh_info[primID][:, 0, :] + np.vstack(u) * mesh_info[primID][:, 1, :] + np.vstack(v) * mesh_info[primID][:, 2, :])

        dists = hits_coord - source_pos

        delay = dists / 343.4

        print('hits_coord: ', hits_coord)
        print('dists: ', dists)
        print('hit: ', self.source_idx, self.output_idx, len(output), len(hits_obj_idx))

#        print('hit: ', self.source_idx, self.output_idx, hits)

#        hits["geomID"]
#        ray_inter = hits["geomID"]
#        primID = hits["primID"][ray_inter]
#        u = hits["u"][ray_inter]
#        v = hits["v"][ray_inter]
#        tfar = hits["tfar"]

#        for hit_idx in range(hits.size):
#            hit = hits[hit_idx]
#            hit["geomID"]
#            ray_inter = hit["geomID"]
#            primID = hit["primID"][ray_inter]
#            u = hit["u"][ray_inter]
#            v = hit["v"][ray_inter]
#            tfar = hit["tfar"]
#        print('hit: ', self.source_idx, self.ouput_idx, hit_idx, ray_inter, primID, u, v , tfar)

    @staticmethod
    @nb.njit(fastmath=True)
    def _find_output_and_obj_idx(raw_obj_idx):
        """
        Optimized version using boolean masks.
        """
        arr = np.asarray(raw_obj_idx, dtype=np.int32)
    
        # Create boolean masks (SIMD operations)
        mask_minus_three = (arr == -3)
        mask_non_negative = (arr >= 0)
    
        # Get indices from masks
        indices_output = np.flatnonzero(mask_minus_three)
        indices_obj_idx = np.flatnonzero(mask_non_negative)

        return indices_output, indices_obj_idx


    @staticmethod
    @nb.njit(fastmath=True)
    def _source_points(n_points: int, source_center: np.ndarray, source_size: float) -> np.ndarray:
        """
        Generate random points uniformly distributed inside a sphere using Marsaglia's method.
        More efficient than rejection sampling.
        """
        points = np.zeros((n_points, 3))
        cx, cy, cz = source_center[0], source_center[1], source_center[2]

        for i in range(n_points):
            # Marsaglia's method for uniform distribution in sphere
            while True:
                # Generate random point on unit disk
                u = 2.0 * np.random.random() - 1.0
                v = 2.0 * np.random.random() - 1.0
                s = u*u + v*v

                if s < 1.0:
                    # Generate random radius with cubic root for uniform volume distribution
                    r = source_size * np.cbrt(np.random.random())

                    # Calculate coordinates
                    sqrt_term = np.sqrt(1.0 - s)
                    x = 2.0 * u * sqrt_term
                    y = 2.0 * v * sqrt_term
                    z = 1.0 - 2.0 * s

                    # Scale and translate
                    points[i, 0] = cx + r * x
                    points[i, 1] = cy + r * y
                    points[i, 2] = cz + r * z
                    break
        return points

    def _generate_isotropic_directions(self, src: np.ndarray, dst: np.ndarray, n_directions: int = 100, seed: int = None) -> List[np.ndarray]:
        """
        Generate random directions with isotropic probability distribution in 4π sr.
 
        Parameters
        ----------
        src : np.array([float, float, float])
            (x, y, z) coordinates of source point
        dst : np.array([float, float, float])
            (x, y, z) coordinates of destination point
        n_directions : int
            Number of random isotropic directions to generate
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        isotropic_dirs : List[np.ndarray]
            List of n_directions unit vectors with isotropic distribution and the normalized unit vector from source to destination
        """

        if seed is not None:
            np.random.seed(seed)

        # Direct direction
        direct_vec = dst - src
        vec_norm = np.linalg.norm(direct_vec)
        if vec_norm < 1e-12:
            raise ValueError("Source and destination are coincident")
        direct_dir = direct_vec / vec_norm

        # Generate isotropic directions
        isotropic_dirs = []

        for _ in range(n_directions):
            # Marsaglia method (1972) for uniform distribution on sphere
            # Generate two uniform random numbers
            while True:
                x1 = np.random.uniform(-1, 1)
                x2 = np.random.uniform(-1, 1)
                s = x1**2 + x2**2
                if s < 1:
                    break

            # Map to sphere surface coordinates
            z = 1 - 2 * s
            factor = 2 * np.sqrt(1 - s)
            x = x1 * factor
            y = x2 * factor

            direction = [x, y, z]
            isotropic_dirs.append(direction)
        isotropic_dirs += [direct_dir.tolist()]
        return isotropic_dirs

    def _generate_initial_directions(self, n_rays: int, source_pos: np.ndarray, output_pos: np.ndarray):
        """Batch-optimized version for maximum performance"""
        n_sources = source_pos.ndim
        n_ray_per_source = int(n_rays / n_sources)
        total_rays = n_sources * n_ray_per_source
        
        # Generate all directions in one batch
        directions = generate_all_directions_batch(total_rays, n_sources, n_ray_per_source, source_pos, output_pos)
        
        return directions
    
#    @staticmethod
#    @nb.njit(parallel=True, fastmath=True, cache=True)
#    def _generate_all_directions_batch(total_rays: int, n_sources: int, n_ray_per_source: int, source_pos: np.ndarray, output_pos: np.ndarray):
#        """Generate all directions in batch with SIMD optimizations"""
#        directions = np.empty((total_rays, 3), dtype=np.float64)
#        
#        # Pre-compute output directions for all sources
#        output_dirs = np.empty((n_sources, 3), dtype=np.float64)
#        for i in range(n_sources):
#            dx = output_pos[0] - source_pos[i, 0]
#            dy = output_pos[1] - source_pos[i, 1]
#            dz = output_pos[2] - source_pos[i, 2]
#            norm = np.sqrt(dx*dx + dy*dy + dz*dz)
#            output_dirs[i, 0] = dx / norm
#            output_dirs[i, 1] = dy / norm
#            output_dirs[i, 2] = dz / norm
#        
#        # Process rays in parallel
#        for i in prange(total_rays):
#            source_idx = i // n_ray_per_source
#            
#            # Thread-safe random using thread ID and iteration
#            thread_id = nb.get_thread_id()
#            seed = thread_id * 1000000 + i
#            
#            # Determine if this is a directed ray (30% probability)
#            # Use deterministic check based on seed
#            if ((seed * 1103515245 + 12345) & 0x7FFFFFFF) % 1000 < 300:
#                # Directed ray with jitter
#                x = output_dirs[source_idx, 0]
#                y = output_dirs[source_idx, 1]
#                z = output_dirs[source_idx, 2]
#                
#                # Generate jitter using fast normal RNG
#                jx, jy, jz = WavePropagator._fast_normal_3(seed)
#                x += jx * 0.1
#                y += jy * 0.1
#                z += jz * 0.1
#                
#                # Normalize
#                norm = np.sqrt(x*x + y*y + z*z)
#                directions[i, 0] = x / norm
#                directions[i, 1] = y / norm
#                directions[i, 2] = z / norm
#            else:
#                # Isotropic ray
#                x, y, z = WavePropagator._fast_isotropic_batch(seed)
#                directions[i, 0] = x
#                directions[i, 1] = y
#                directions[i, 2] = z
#        
#        return directions
#    
#    @staticmethod
#    @nb.njit(fastmath=True, inline='always')
#    def _fast_isotropic_batch(seed):
#        """Ultra-fast isotropic direction using rejection sampling"""
#        # Xorshift for speed
#        state = np.uint64(seed)
#        
#        while True:
#            state ^= state << 13
#            state ^= state >> 7
#            state ^= state << 17
#            
#            # Get two random numbers in [-1, 1]
#            # Using bit manipulation for speed
#            u1 = ((state & 0xFFFFFFFF) / 4294967295.0) * 2.0 - 1.0
#            state ^= state << 13
#            state ^= state >> 7
#            state ^= state << 17
#            u2 = ((state & 0xFFFFFFFF) / 4294967295.0) * 2.0 - 1.0
#            
#            s = u1*u1 + u2*u2
#            if s < 1.0 and s > 0.0:  # Avoid division by zero
#                sqrt_term = np.sqrt(1.0 - s)
#                x = 2.0 * u1 * sqrt_term
#                y = 2.0 * u2 * sqrt_term
#                z = 1.0 - 2.0 * s
#                return x, y, z
#    
#    @staticmethod
#    @nb.njit(fastmath=True, inline='always')
#    def _fast_normal_3(seed):
#        """Generate 3 normal random numbers - optimized version"""
#        # Xorshift RNG
#        state = np.uint64(seed)
#        
#        # Generate 6 uniform random numbers for 3 normals
#        uniforms = np.empty(6, dtype=np.float64)
#        for j in range(6):
#            state ^= state << 13
#            state ^= state >> 7
#            state ^= state << 17
#            uniforms[j] = (state & 0xFFFFFFFF) / 4294967295.0
#        
#        # Box-Muller transform
#        r0 = np.sqrt(-2.0 * np.log(uniforms[0]))
#        theta0 = 2.0 * np.pi * uniforms[1]
#        z0 = r0 * np.cos(theta0)
#        z1 = r0 * np.sin(theta0)
#        
#        r1 = np.sqrt(-2.0 * np.log(uniforms[2]))
#        theta1 = 2.0 * np.pi * uniforms[3]
#        z2 = r1 * np.cos(theta1)
#        
#        return z0, z1, z2
#
    def do_not_run(self):
        # Main propagation loop
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

    def xxxx_generate_initial_directions(self, n_rays: int, source_pos: np.ndarray, output_pos: np.ndarray):
        """Generate initial rays with importance sampling"""
        # Importance sampling: more rays towards output
        directions = []
        n_ray = int(n_rays / source_pos.shape[0])
        for source_idx in range(source_pos.shape[0]):
            for i in range(n_ray):
                # Blend between isotropic and directed sampling
                if np.random.random() < 0.3:  # 30% directed rays
                    direction = output_pos - source_pos[source_idx]
                    direction = direction / np.linalg.norm(direction)
                    # Add some jitter
                    direction += np.random.normal(0, 0.1, 3)
                    direction = direction / np.linalg.norm(direction)
                else:
                    # Isotropic direction
                    direction = self._random_isotropic_direction()
                directions.append(direction)
            return directions
    
    @staticmethod
    @nb.njit(fastmath=True)
    def xxxx_random_isotropic_direction():
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
