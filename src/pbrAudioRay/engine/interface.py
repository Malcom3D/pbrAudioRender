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

import dask
from dask import delayed, compute
import numpy as np
import trimesh
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.ray_data import RayData
from ..engine.ray_tracer import RayTracer
from ..engine.interfaces import AbsorptionInterface, ReflectionInterface, RefractionInterface, ScatteringInterface, DiffractionInterface

@dataclass
class InterfaceManager:
    """Main interface manager handling all boundary interactions."""
    entity_manager: EntityManager
    ray_idx: int = 0

    def __post_init__(self):
        config = self.entity_manager.get('config')

        # Get sample rate
        self.sample_rate = config.system.sample_rate

        self.ray_tracer = RayTracer(self.entity_manager)
        self.absorption = AbsorptionInterface(self.entity_manager)
        self.reflection = ReflectionInterface(self.entity_manager)
        self.refraction = RefractionInterface(self.entity_manager)
        self.scattering = ScatteringInterface(self.entity_manager)
        self.diffraction = DiffractionInterface(self.entity_manager)
        
#        self.interaction_threshold = config.interface.interaction_threshold # it's neeeded for FDTD?
        self.min_impedance_ratio = config.interface.min_impedance_ratio
        self.max_impedance_ratio = config.interface.max_impedance_ratio
    
    def compute(self, frame_idx: int, source_pos: np.ndarray, source_rot: np.ndarray, output_pos: np.ndarray, output_rot: np.ndarray, scene_meshes: List[trimesh.Trimesh], scene_meshes_ids: List[int]):
        config = self.entity_manager.get('config')

        # Get rays config
        number_of_rays = config.system.number_of_rays
        direction_seed = config.system.direction_seed

        # Get Frequency bands for impulse response
        frequency_bands = self.entity_manager.get('frequency_bands')

        # Compute direct and reverse isotropic directions
        direct_isotropic_directions = self._generate_isotropic_directions(source_pos, output_pos, number_of_rays, direction_seed)
        reverse_isotropic_directions = self._generate_isotropic_directions(output_pos, source_pos, number_of_rays, direction_seed)

        direct_task, reverse_task = ([] for _ in range(2))
        total_bands = len(frequency_bands.get_bands())
        # Trace direct ray path (source to output)
        for bands_idx in range(total_bands):
            for direction in direct_isotropic_directions:
                direct_task += [self._trace_path(source_pos, direction, bands_idx, scene_meshes, scene_meshes_ids)]
                self.ray_idx += 1
        direct_rays = compute(*direct_task)

        # Trace reverse ray path (output to source)
        for bands_idx in range(total_bands):
            for direction in reverse_isotropic_directions:
                reverse_task += [self._trace_path(output_pos, direction, bands_idx, scene_meshes, scene_meshes_ids)]
                self.ray_idx += 1
        reverse_rays = compute(*reverse_task)

        print('direct_rays: ', direct_rays)
        print('reverse_rays: ', reverse_rays)
        print('results: ', len(direct_rays), len(reverse_rays))

    @delayed
    def _trace_path(self, src: np.ndarray, direction: np.ndarray, bands_idx: int, scene_meshes: List[trimesh.Trimesh], scene_meshes_ids: List[int]):
        """Trace direct line-of-sight path."""
        hit = self.ray_tracer.intersect_ray(src, direction, scene_meshes, scene_meshes_ids)
        # Create ray data
        if not hit['object_idx'] == -1: # ray hit the AcousticDomain
            if hit['hit'] == False:
                length = dist
                point = None
                normal = None
            else:
                length = hit['distance']
                point = hit['point']
                normal = hit['normal']

            return RayData(
                origin=src,
                direction=direction,
                ray_idx=self.ray_idx,
                bands_idx=bands_idx,
                length=length,
                energy=1.0,  # initial energy
                reflection_count=0,
                path=[src, dst],
                hit=hit['hit'],
                object_idx=hit['object_idx'],
                point=point,
                normal=normal
            )

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
        
            direction = np.array([x, y, z])
            isotropic_dirs.append(direction)
        isotropic_dirs += [direct_dir]
        return isotropic_dirs





    def compute_interaction(self, ray, hit_info, source_pos, listener_pos):
        """Apply all interactions at a hit point."""
        # Get object config
        obj_idx = hit_info['object_idx']
        objs_config = self.entity_manager.get('objects')
        for c_idx in objs_config.keys():
            if objs_config[c_idx].idx == obj_idx:
                obj_config = objs_config[c_idx]
        
        # Get incident angle
        incident_dir = ray.direction
        normal = hit_info['normal']
        # Ensure normal points towards the ray
        if np.dot(incident_dir, normal) > 0:
            normal = -normal  # flip so it points outward from object surface
        angle_incident = np.arccos(np.dot(incident_dir, normal) / (np.linalg.norm(incident_dir) * np.linalg.norm(normal)))
        
        # Get acoustic shader for the object
        shader = obj_config.acoustic_shader
        if shader is None:
            # Use default properties from main medium? For now, no interaction
            return ray
        
        # Compute absorption, reflection, etc.
        # For frequency-dependent, we need to pass frequency band. We'll do it later.
        # For now, just simple scaling.
        # Absorption: reduce energy
        if shader.acoustic_properties and shader.acoustic_properties.absorption:
            # Get absorption coefficient at given frequency (we'll use average over bands)
            # For simplicity, use average absorption over all frequencies
            coeffs = shader.acoustic_properties.absorption.get_avg_coeffs()
            # Apply absorption: energy *= (1 - coeff)
            ray.energy *= (1 - coeffs)
        
        # Reflection: change direction
        if shader.acoustic_properties and shader.acoustic_properties.reflection:
            # Compute reflection direction
            reflect_dir = incident_dir - 2 * np.dot(incident_dir, normal) * normal
            reflect_dir = reflect_dir / np.linalg.norm(reflect_dir)
            ray.direction = reflect_dir
            # Apply reflection coefficient
            coeffs = shader.acoustic_properties.reflection.get_avg_coeffs()
            ray.energy *= coeffs
        
        # Refraction: if entering different medium
        # We need to know sound speed of object vs main medium
        # For now, skip
        
        # Scattering: add random perturbation to direction
        if shader.acoustic_properties and shader.acoustic_properties.scattering:
            # Simple scattering: add random component
            scatter_strength = shader.acoustic_properties.scattering.get_avg_coeffs()
            # Perturb direction randomly
            random_dir = np.random.randn(3)
            random_dir = random_dir / np.linalg.norm(random_dir)
            ray.direction = (1 - scatter_strength) * reflect_dir + scatter_strength * random_dir
            ray.direction = ray.direction / np.linalg.norm(ray.direction)
        
        # Update ray origin to hit point
        ray.origin = hit_info['point']
        ray.path.append(hit_info['point'])
        ray.reflection_count += 1
        
        return ray
