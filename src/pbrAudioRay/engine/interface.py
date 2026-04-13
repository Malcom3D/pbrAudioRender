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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..engine.interfaces import AbsorptionInterface, ReflectionInterface, RefractionInterface, ScatteringInterface, DiffractionInterface

@dataclass
class InterfaceManager:
    """Main interface manager handling all boundary interactions."""
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.absorption = AbsorptionInterface(self.entity_manager)
        self.reflection = ReflectionInterface(self.entity_manager)
        self.refraction = RefractionInterface(self.entity_manager)
        self.scattering = ScatteringInterface(self.entity_manager)
        self.diffraction = DiffractionInterface(self.entity_manager)
        
        self.interaction_threshold = config.interface.interaction_threshold
        self.min_impedance_ratio = config.interface.min_impedance_ratio
        self.max_impedance_ratio = config.interface.max_impedance_ratio
    
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
