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

import os
import json
import numpy as np
import numba as nb
from dask import delayed, compute
from typing import Tuple, Optional, List, Any, Dict
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.ray_data import RayData
from ..lib.functions import _compute_rayleigh_damping

#from .interfaces import AbsorptionInterface, ReflectionInterface, TransmissionInterface, ScatteringInterface, DiffractionInterface
from .interfaces.absorption import AbsorptionInterface
from .interfaces.reflection import ReflectionInterface
from .interfaces.scattering import ScatteringInterface
from .interfaces.transmission import TransmissionInterface

@dataclass
class InterfaceManager:
    """
    Handle rays interaction with objects boundaries
    """
    entity_manager: EntityManager
    geometry_data: Any
    material_properties: Any
    medium_properties: Any
    ray_data: Any
    output_data: Any
    frame_idx: int
    recursion_idx: int = 0
    
    def __post_init__(self):
        config = self.entity_manager.get('config')

        self.absorption_interface = AbsorptionInterface(self.entity_manager)
        self.reflection_interface = ReflectionInterface(self.entity_manager)
        self.scattering_interface = ScatteringInterface(self.entity_manager)
        self.transmission_interface = TransmissionInterface(self.entity_manager)


    def compute(self, res: Dict[str, np.ndarray], ray_inter: np.ndarray):
        """
        Process ray hits and deliver data to interface subclasses.
        
        Args:
            hits: Dictionary containing ray intersection results from Embree run
            
        Returns:
            Tuple of (next_source_positions, next_directions, bands_idx, ray_data)
        """
        config = self.entity_manager.get('config')

        primID = res["primID"][ray_inter]
        u = res["u"][ray_inter]
        v = res["v"][ray_inter]
        w = 1 - u - v

        a = self.geometry_data.mesh_info[primID][:, 0, :]
        b = self.geometry_data.mesh_info[primID][:, 1, :]
        c = self.geometry_data.mesh_info[primID][:, 2, :]

        inters = (np.vstack(w) * a + np.vstack(u) * b + np.vstack(v) * c)

        # Save ray data
        # ToDo: only for the first frame.....
        if config.system.view_ray and self.ray_data.bands_idx == 0:
            self._save_ray_data(self.ray_data.origins, inters)

        # Filter rays that intersect
        self._filter_intersected_rays(ray_inter, primID, inters)

    def _save_ray_data(self, origins: np.ndarray, hit_points: np.ndarray):
        """Save ray data to JSON file."""
        data_dict = {
            'origins': origins.tolist(),
            'hit_points': hit_points.tolist()
        }

        os.makedirs('ray_datas', exist_ok=True)
        filepath = f"ray_datas/embreex_{self.recursion_idx:04}.json"

        with open(filepath, 'w') as f:
            json.dump(data_dict, f, indent=2)

    def _filter_intersected_rays(self, ray_inter: np.ndarray, primID: np.ndarray, inters: np.ndarray):
        """Filter and process intersected rays."""
        # Filter all ray data
        self.ray_data.origins = self.ray_data.origins[ray_inter]
        self.ray_data.directions = self.ray_data.directions[ray_inter]
        self.ray_data.energies = self.ray_data.energies[ray_inter]
        self.ray_data.phases = self.ray_data.phases[ray_inter]
        self.ray_data.delay = self.ray_data.delay[ray_inter]

        # Compute path length
        path_length = np.sqrt(np.sum((inters - self.ray_data.origins)**2, axis=1)).reshape(-1, 1)

        # Update medium properties
        self._update_medium_properties(path_length)

        # Get object indices and filter
        self.hits_obj_idx = self.geometry_data.scene_info[primID]
        intersect_mask = self.hits_obj_idx >= 0

        # Collect output data
        self._collect_output_data(intersect_mask, path_length)

        # Continue with remaining rays
        if np.any(intersect_mask):
            self._continue_tracing(inters, primID, path_length, intersect_mask)

    def _update_medium_properties(self, path_length: np.ndarray):
        """Update medium attenuation and phase shift."""
        frequency_bands = self.entity_manager.get('frequency_bands')
        n_bands = len(frequency_bands.get_bands())
        bands_idx = self.ray_data.bands_idx
        n_rays = self.ray_data.origins.shape[0]

        # Default medium properties (air)
        self.medium_objs = np.full((n_rays, 1), [-1], dtype=np.int32)
        medium_speed = np.full((n_rays, 1), self.medium_properties.speed, dtype=np.float32)
        medium_alpha = np.full((n_rays, n_bands), self.medium_properties.alpha, dtype=np.float32)
        medium_beta = np.full((n_rays, n_bands), self.medium_properties.beta, dtype=np.float32)

        # Check for objects containing origins
        config = self.entity_manager.get('config')
        objects = self.entity_manager.get('objects')
        for key in objects:
            mesh = objects[key].get_mesh(self.frame_idx)
            for obj_config in config.objects:
                if obj_config.idx == objects[key].obj_idx:
                    medium_mask = mesh.contains(self.ray_data.origins)
                    if np.any(medium_mask):
                        sound_speed = obj_config.acoustic_shader.sound_speed
                        density = obj_config.acoustic_shader.density
                        young_modulus = obj_config.acoustic_shader.young_modulus
                        poisson_ratio = obj_config.acoustic_shader.poisson_ratio
                        damping = obj_config.acoustic_shader.damping

                        alpha, beta = self._compute_object_coefficients(sound_speed, density, young_modulus, poisson_ratio, damping)

                        self.medium_objs[medium_mask] = obj_config.idx
                        medium_speed[medium_mask] = sound_speed
                        medium_alpha[medium_mask] = np.full((1,n_bands), [alpha], dtype=np.float32)
                        medium_beta[medium_mask] = np.full((1,n_bands), [beta], dtype=np.float32)

        # Apply medium attenuation
        attenuation = np.exp(-medium_alpha * path_length)
        self.ray_data.energies *= attenuation[:,bands_idx].reshape(-1,1)

        # Apply phase shift
        phase_shift = path_length * medium_beta
        self.ray_data.phases = (self.ray_data.phases + phase_shift[:,bands_idx].reshape(-1,1)) % (2 * np.pi)

        # Update delay
        new_delay = path_length / medium_speed
        self.ray_data.delay += new_delay

    def _compute_object_coefficients(self, c: float, rho: float, E: float, nu: float, damping: float):
        """Calculate medium attenuation coefficient and phase shift for objects."""
        frequency_bands = self.entity_manager.get('frequency_bands')

        min_freq, max_freq = frequency_bands.get_bands()[self.ray_data.bands_idx]
        alpha, beta = _compute_rayleigh_damping(min_freq, max_freq, damping)

        freqs = min_freq + (max_freq - min_freq)/2
        omega = 2 * np.pi * freqs

        K = E / (3 * (1 - 2 * nu))
        G = E / (2 * (1 + nu)) 
        Z = rho * c

        is_solid = G > 0.1 * E

        alpha_attenuation = (alpha / (2 * c)) + (beta * omega**2 / (2 * c))

        if not is_solid:
            viscosity = 1.8e-5 if rho < 100 else 1e-3
            alpha_viscous = (2 * omega**2 * viscosity) / (3 * rho * c**3)
        else:
            alpha_viscous = 0

        alpha_attenuation = alpha_attenuation + alpha_viscous
        phase_shift = omega / c

        return alpha_attenuation, phase_shift

    def _continue_tracing(self, inters: np.ndarray, primID: np.ndarray, path_length: np.ndarray, intersect_mask: np.ndarray):
        """Continue ray tracing for rays that hit objects."""
        # Filter for rays that hit objects
        inters = inters[intersect_mask]
        path_length = path_length[intersect_mask]

        # Compute normals
        a = self.geometry_data.mesh_info[primID][:, 0, :]
        b = self.geometry_data.mesh_info[primID][:, 1, :]
        c = self.geometry_data.mesh_info[primID][:, 2, :]
        
        normals = np.cross(b - a, c - a)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals[intersect_mask]
            
        # Filter remaining data
        self.ray_data.origins = self.ray_data.origins[intersect_mask]
        self.ray_data.directions = self.ray_data.directions[intersect_mask]
        self.ray_data.energies = self.ray_data.energies[intersect_mask]
        self.ray_data.phases = self.ray_data.phases[intersect_mask]
        self.ray_data.delay = self.ray_data.delay[intersect_mask]
            
        # Get material properties for intersected faces
        self._apply_material_properties(primID, inters, intersect_mask, normals)

    def _apply_material_properties(self, primID: np.ndarray, inters: np.ndarray, intersect_mask: np.ndarray, normals: np.ndarray):
        """Apply material properties at intersection points."""
        config = self.entity_manager.get('config')
        enable_absorption = config.interface.enable_absorption
        enable_reflection = config.interface.enable_reflection
        enable_scattering = config.interface.enable_scattering
        enable_transmission = config.interface.enable_transmission

        primID_filtered = primID[intersect_mask]
        self.medium_objs = self.medium_objs[intersect_mask]

        # Compute new origins
        new_origins = inters + (0.01 * normals)
        new_origins = new_origins.astype(np.float32)

        # Get material properties
        if enable_absorption:
            absorbed_energies = self.absorption_interface.compute(self.material_properties, primID_filtered, normals, self.ray_data)
        else:
            absorbed_energies = self.ray_data.energies

        if enable_reflection:
            reflected_energies, reflected_phases, reflected_directions = self.reflection_interface.compute(self.material_properties, primID_filtered, normals, self.ray_data, new_origins)
            reflected_origins = new_origins
            reflected_delay = self.ray_data.delay
        else:
            reflected_origins = np.zeros((0,3), dtype=np.float32)
            reflected_directions = np.zeros((0,3), dtype=np.float32)
            reflected_energies = np.zeros((0,1), dtype=np.float32)
            reflected_phases = np.zeros((0,1), dtype=np.float32)
            reflected_delay = np.zeros((0,1), dtype=np.float32)

        if enable_transmission:
            transmission_data = self.transmission_interface.compute(self.medium_objs, self.hits_obj_idx, self.material_properties, inters, primID_filtered, normals, self.ray_data, absorbed_energies, self.frame_idx)
        else:
            transmission_data = {
                'origins': np.zeros((0, 3), dtype=np.float32),
                'directions': np.zeros((0, 3), dtype=np.float32),
                'normals': np.zeros((0, 3), dtype=np.float32),
                'energies': np.zeros((0, 1), dtype=np.float32),
                'phases': np.zeros((0, 1), dtype=np.float32),
                'delay': np.zeros((0, 1), dtype=np.float32)
            }

        if enable_scattering:
            scattered_data = self.scattering_interface.compute(self.material_properties, primID_filtered, normals, self.ray_data, new_origins)
        else:
            scattered_data = {
                'origins': np.zeros((0, 3), dtype=np.float32),
                'directions': np.zeros((0, 3), dtype=np.float32),
                'normals': np.zeros((0, 3), dtype=np.float32),
                'energies': np.zeros((0, 1), dtype=np.float32),
                'phases': np.zeros((0, 1), dtype=np.float32),
                'delay': np.zeros((0, 1), dtype=np.float32)
            }

        # Energy conservation check
        total_out = np.sum(absorbed_energies) + np.sum(reflected_energies) + np.sum(scattered_data['energies'])
        scale = np.sum(self.ray_data.energies) / total_out
        absorbed_energies *= scale
        reflected_energies *= scale
        transmission_data['energies'] *= scale
        scattered_data['energies'] *= scale

        # Combine reflected, transmitted and scattered rays
        self.ray_data.origins = np.concatenate((reflected_origins, scattered_data['origins'], transmission_data['origins']), axis=0)
        self.ray_data.directions = np.concatenate((reflected_directions, scattered_data['directions'], transmission_data['directions']), axis=0)
        self.ray_data.energies = np.concatenate((reflected_energies, scattered_data['energies'], transmission_data['energies']), axis=0)
        self.ray_data.phases = np.concatenate((reflected_phases, scattered_data['phases'], transmission_data['phases']), axis=0)
        self.ray_data.delay = np.concatenate((reflected_delay, scattered_data['delay'], transmission_data['delay']), axis=0)

    def _collect_output_data(self, intersect_mask: np.ndarray, path_length: np.ndarray):
        """Collect rays that reached output destinations."""
        output_mask = self.hits_obj_idx <= -3

        if np.any(output_mask):
            self.output_data.energies = np.append(self.output_data.energies, self.ray_data.energies[output_mask], axis=0).astype(np.float32)
            self.output_data.phases = np.append(self.output_data.phases, self.ray_data.phases[output_mask], axis=0).astype(np.float32)
            self.output_data.delay = np.append(self.output_data.delay, self.ray_data.delay[output_mask], axis=0).astype(np.float32)
            self.output_data.origins = np.append(self.output_data.origins, self.ray_data.origins[output_mask], axis=0).astype(np.float32)
            self.output_data.directions = np.append(self.output_data.directions, self.ray_data.directions[output_mask], axis=0).astype(np.float32)
            print(f'Output: {np.count_nonzero(output_mask)}, 'f'{self.output_data.energies.shape[0]}')
