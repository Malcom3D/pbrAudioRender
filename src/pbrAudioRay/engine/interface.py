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
    acoustic_rays: Any

    def __post_init__(self):
        self.absorption = AbsorptionInterface(self.acoustic_rays)
        self.reflection = ReflectionInterface(self.acoustic_rays)
        self.refraction = RefractionInterface(self.acoustic_rays)
        self.scattering = ScatteringInterface(self.acoustic_rays)
        self.diffraction = DiffractionInterface(self.acoustic_rays)
        
#        self.interaction_threshold = config.interface.interaction_threshold # it's neeeded for FDTD?
        self.min_impedance_ratio = config.interface.min_impedance_ratio
        self.max_impedance_ratio = config.interface.max_impedance_ratio
    
    def compute(self, hit_points: np.ndarray, normals: np.ndarray, absorption: np.ndarray, reflection: np.ndarray, refraction: np.ndarray, scattering: np.ndarray, hit_objects: np.ndarray, bands_idx: int, medium_idx: int = None):

        # if medium_idx is present this is the first acoustic rays propagation loop
        if not medium_idx:
            current_medium_idx = medium_idx

        """Compute angle influence for coefficients computation"""
        sources, directions = self.acoustic_rays.get_od(bands_idx)
        incident_angles = self._compute_incident_angles(directions, hit_points, normals)
        angle_factors = np.cos(incident_angle) if incident_angle < np.pi/2 else 0.0

        """Compute absorption"""
        self.absorption.compute(hit_points, absorption, hit_objects, angle_factors)
        
        absorption_coeffs = absorption * angle_factor

        """Compute reflection coefficients"""

        """Compute reflected rays directions"""
        next_directions = self.reflect_rays(directions, normals)


    @staticmethod
    @njit(nogil=True, fastmath=True, cache=True)
    def reflect_rays(directions: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Reflect ray directions using vectorized operations"""
        dot = np.sum(directions * normals, axis=1)
        return directions - 2 * dot[:, np.newaxis] * normals

    def _compute_incident_angles(ray_directions: np.ndarray, intersection_points: np.ndarray, normals: np.ndarray):
        """
        Compute incident angles for rays intersecting with surfaces.
    
        Parameters:
        -----------
        ray_directions : numpy.ndarray
            Direction vectors of rays, shape (n_rays, 3)
        intersection_points : numpy.ndarray
            Points of intersection, shape (n_rays, 3)
        normals : numpy.ndarray
            Surface normals at intersection points, shape (n_rays, 3)
    
        Returns:
        --------
        incident_angles : numpy.ndarray
            Incident angles in radians, shape (n_rays,)
        """
        # Normalize the ray direction vectors
        ray_directions_normalized = ray_directions / np.linalg.norm(ray_directions, axis=1, keepdims=True)
    
        # Normalize the normals
#        normals_normalized = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        normals_normalized = normals
    
        # Compute the dot product between ray direction and normal
        # Note: We use the negative of ray direction since incident angle is measured
        # between the incoming ray and the surface normal
        dot_products = np.sum(-ray_directions_normalized * normals_normalized, axis=1)
    
        # Clamp dot products to [-1, 1] to avoid numerical issues
        dot_products = np.clip(dot_products, -1.0, 1.0)
    
        # Compute incident angles (arccos of dot product)
        incident_angles = np.arccos(dot_products)
    
        return incident_angles

#    def compute_interaction(self, ray, hit_info, source_pos, listener_pos):
#        """Apply all interactions at a hit point."""
#        # Get object config
#        obj_idx = hit_info['object_idx']
#        objs_config = self.entity_manager.get('objects')
#        for c_idx in objs_config.keys():
#            if objs_config[c_idx].idx == obj_idx:
#                obj_config = objs_config[c_idx]
#        
#        # Get incident angle
#        incident_dir = ray.direction
#        normal = hit_info['normal']
#        # Ensure normal points towards the ray
#        if np.dot(incident_dir, normal) > 0:
#            normal = -normal  # flip so it points outward from object surface
#        angle_incident = np.arccos(np.dot(incident_dir, normal) / (np.linalg.norm(incident_dir) * np.linalg.norm(normal)))
#        
#        # Get acoustic shader for the object
#        shader = obj_config.acoustic_shader
#        if shader is None:
#            # Use default properties from main medium? For now, no interaction
#            return ray
#        
#        # Compute absorption, reflection, etc.
#        # For frequency-dependent, we need to pass frequency band. We'll do it later.
#        # For now, just simple scaling.
#        # Absorption: reduce energy
##        if shader.acoustic_properties and shader.acoustic_properties.absorption:
#            # Get absorption coefficient at given frequency (we'll use average over bands)
#            # For simplicity, use average absorption over all frequencies
#            coeffs = shader.acoustic_properties.absorption.get_avg_coeffs()
#            # Apply absorption: energy *= (1 - coeff)
#            ray.energy *= (1 - coeffs)
#        
#        # Reflection: change direction
#        if shader.acoustic_properties and shader.acoustic_properties.reflection:
#            # Compute reflection direction
#            reflect_dir = incident_dir - 2 * np.dot(incident_dir, normal) * normal
#            reflect_dir = reflect_dir / np.linalg.norm(reflect_dir)
#            ray.direction = reflect_dir
#            # Apply reflection coefficient
#            coeffs = shader.acoustic_properties.reflection.get_avg_coeffs()
#            ray.energy *= coeffs
#        
#        # Refraction: if entering different medium
#        # We need to know sound speed of object vs main medium
#        # For now, skip
#        
#        # Scattering: add random perturbation to direction
#        if shader.acoustic_properties and shader.acoustic_properties.scattering:
#            # Simple scattering: add random component
#            scatter_strength = shader.acoustic_properties.scattering.get_avg_coeffs()
#            # Perturb direction randomly
#            random_dir = np.random.randn(3)
#            random_dir = random_dir / np.linalg.norm(random_dir)
#            ray.direction = (1 - scatter_strength) * reflect_dir + scatter_strength * random_dir
#            ray.direction = ray.direction / np.linalg.norm(ray.direction)
#        
#        # Update ray origin to hit point
#        ray.origin = hit_info['point']
#        ray.path.append(hit_info['point'])
#        ray.reflection_count += 1
#        
###        return ray
