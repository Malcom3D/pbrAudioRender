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
from dask import delayed, compute
from typing import Tuple, Optional, List, Any, Dict
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from .interface import InterfaceManager

@dataclass
class DiffPathTracer:
    """
    Implements differentiable path tracing for acoustic rendering.
    Based on: https://pub.dega-akustik.de/DAGA_2024/files/upload/paper/489.pdf
    """
    entity_manager: EntityManager
    acoustic_scene: Any  # AcousticScene
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.max_interactions = config.wave_propagation.max_interactions

        self.n_bands = len(self.entity_manager.get('frequency_bands').get_bands())

        # Initialize counters
        self.ray_count = 0
        self.current_interactions = 0

    @delayed
    def compute(self, hits: Dict, bands_idx: int, ray_data: Any) -> Tuple[np.ndarray, np.ndarray, int, Any]:
        """
        Process ray hits and generate new rays for next bounce.
        
        Args:
            hits: Dictionary containing ray intersection results from Embree
            
        Returns:
            Tuple of (next_source_positions, next_directions)
        """

        if self.current_interactions == self.max_interactions:
            next_positions = np.array([])
            next_directions = np.array([])

#        self.interface = InterfaceManager(self.entity_manager, self.acoustic_scene, ray_data)

        # Get scene data
#        mesh_info = self.acoustic_scene.mesh_info
#        scene_info = self.acoustic_scene.scene_info
#
#        # get objects material properties array
#        sound_speed = self.acoustic_scene.sound_speed
#        density = self.acoustic_scene.density
#        roughness = self.acoustic_scene.roughness
#        absorption_coeffs = self.acoustic_scene.absorption_coeffs.reshape(mesh_info.shape[0], self.n_bands)
#        absorption_phases = self.acoustic_scene.absorption_phases.reshape(mesh_info.shape[0], self.n_bands)
#        refraction_coeffs = self.acoustic_scene.refraction_coeffs.reshape(mesh_info.shape[0], self.n_bands)
#        refraction_phases = self.acoustic_scene.refraction_phases.reshape(mesh_info.shape[0], self.n_bands)
#        reflection_coeffs = self.acoustic_scene.reflection_coeffs.reshape(mesh_info.shape[0], self.n_bands)
#        reflection_phases = self.acoustic_scene.reflection_phases.reshape(mesh_info.shape[0], self.n_bands)
#        scattering_coeffs = self.acoustic_scene.scattering_coeffs.reshape(mesh_info.shape[0], self.n_bands)
#        scattering_phases = self.acoustic_scene.scattering_phases.reshape(mesh_info.shape[0], self.n_bands)

        source_pos = self.acoustic_scene.aso_pos[0]
        output_pos = self.acoustic_scene.aso_pos[1]

        source_medium = self.acoustic_scene.aso_medium[0]
        output_medium = self.acoustic_scene.aso_medium[1]

        recursions = ray_data.recursions
        recursion_idx = np.unique(recursions)[-1]
        origins = ray_data.origins[recursions == recursion_idx]

        geom_ids = hits["geomID"] >= 0
        prim_ids = hits["primID"][geom_ids]

        # Initialize arrays for next ray data
        next_positions = []
        next_directions = []

        u = hits["u"][prim_ids]
        v = hits["v"][prim_ids]
        w = 1 - u - v

#        triangle = mesh_info[prim_ids]
        triangle = self.acoustic_scene.get_mesh_info(mask=prim_ids)
        a = triangle[:, 0, :]
        b = triangle[:, 1, :]
        c = triangle[:, 2, :] 
        
        # Compute hit point using barycentric coordinates
        hit_points = (np.vstack(w) * a + np.vstack(u) * b + np.vstack(v) * c)

        # Get main medium properties
        ac_sound_speed = self.acoustic_scene.ac_sound_speed
        ac_density = self.acoustic_scene.ac_density
        ac_attenuation = self.acoustic_scene.ac_attenuation[bands_idx]

        # Compute traveled path length
#        path_length = np.sqrt(np.sum((hit_points - origins[prim_ids])**2, axis=1))
        path_length = np.sqrt(np.sum((hit_points - origins)**2, axis=1)).reshape(-1,1)

        # Compute dalay in main medium
#        sound_c = sound_speed[intersect_mask][:,:,bands_idx]
        delay = path_length * ac_sound_speed # all path are on the acoustic domain

        # Compute energy attenuation and phase shift after traveled path using exponential decay
        # E = E0 * exp(-alpha * distance)
        # where alpha is in nepers/m
        initial_energy = ray_data.energies[recursion_idx]
        attenuation = np.exp(-ac_attenuation[0] * path_length)
        rays_energies = initial_energy * attenuation

        # Calculate phase shift
        # Phase = beta * distance (in radians)
        initial_phase = ray_data.phases[recursion_idx]
        phase_shift = ac_attenuation[1] * path_length
        rays_phases = (initial_phase + phase_shift) % (2 * np.pi)
    
        # Wrap phase to [-π, π] range for better numerical representation
        rays_phases = np.mod(rays_phases + np.pi, 2 * np.pi) - np.pi

        # filter rays_energies and rays_phases for output and objects hits
#        hit_obj_idx = scene_info[prim_ids]
        hit_obj_idx = self.acoustic_scene.get_scene_info(mask=prim_ids)
#        output_mask = hit_obj_idx == -3
        intersect_mask = hit_obj_idx >= 0

        rays_energies = rays_energies.reshape(-1,1)
        rays_energies = rays_energies.reshape(-1,1)

        rays_energies_output = rays_energies
        rays_phases_output = rays_phases

        rays_energies = rays_energies[prim_ids]
        rays_energies = rays_energies[intersect_mask]
        rays_phases = rays_phases[prim_ids]
        rays_phases = rays_phases[intersect_mask]

        # Get material properties
        abs_coeffs, abs_phases = self.acoustic_scene.get_absorption(mask=prim_ids, bands_idx=bands_idx)
        refl_coeffs, refl_phases = self.acoustic_scene.get_reflection(mask=prim_ids, bands_idx=bands_idx)
        refr_coeffs, refr_phases = self.acoustic_scene.get_refraction(mask=prim_ids, bands_idx=bands_idx)
        scat_coeffs, scat_phases = self.acoustic_scene.get_scattering(mask=prim_ids, bands_idx=bands_idx)

        # Compute triangle normals using np.cross with broadcasting
        normals = np.cross(b-a, c-a)
            
        # Normalize (avoid division by zero)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        # Compute incident angles and reflected directions
        directions = ray_data.directions[recursion_idx]
        incident_angles, reflected_directions = self._compute_reflections(directions, normals)

        # Compute scattered directions (random direction in hemisphere)
        scattered_directions = self._random_hemisphere_directions(normals)

        # Compute intersection absorption energies (no phase shift)
        angle_factor = np.cos(incident_angles) if incident_angles < np.pi/2 else 1
        absorbed_energy = rays_energies * abs_coeffs * angle_factor

        # Compute intersection reflection energies and phase shift
        reflected_energy = rays_energies * refl_coeffs
        reflected_phase = rays_phases + refl_phases % (2 * np.pi)

        # Compute intersection scattering energies and phase shift
        roughness_factor = self.acoustic_scene.get_roughness(mask=prim_ids)
        scattered_energy = rays_energies * scat_coeffs * roughness_factor / np.max(scattered_directions.shape[0], 1e-10)
        scattered_phase = rays_phases + scat_phases % (2 * np.pi)

        # Energy conservation check
        total_out = reflected_energy + scattered_energy + absorbed_energy
        if abs(total_out - rays_energies) > 1e-10:
            # Normalize to ensure energy conservation
            scale = rays_energies / total_out
            reflected_energy *= scale
            scattered_energy *= scale
            absorbed_energy *= scale

        # Append hit_points for origins, directions, energies and phases
        appended_origins = np.append(hit_points, hit_points, axis=0)
        appended_directions = np.append(reflected_directions, scattered_directions, axis=0)
        appended_energies = np.append(reflected_energy, scattered_energy, axis=0)
        appended_phases = np.append(reflected_phase, scattered_phase, axis=0)
        
        # Filter direction, hit_points and phases on energy termination
        termination_energy = 1e-6
        termination_mask = appended_energies > termination_energy
        new_energies = appended_energies[termination_mask]
        new_phases = appended_phases[termination_mask]
        new_directions = appended_directions[termination_mask.reshape(-1,)]
        new_origins = appended_origins[termination_mask.reshape(-1,)]

        # Complete recursion data in RayData storage
        ray_data.add_data(recursion_idx=recursion_idx, n_rays=hit_points.shape[0], hits_coords=hit_points, path_length=path_length, delay=delay, rays_energies_output=rays_energies_output, rays_phases_output=rays_phases_output, hit_obj_idx=hit_obj_idx)

        # Register new recursion data in RayData storage
        recursion_idx += 1
        ray_data.add_data(recursion_idx=recursion_idx, n_rays=new_origins.shape[0], origins=new_origins, new_directions=directions, energies=new_energies, phases=new_phases) 

        self.current_interactions += 1
        return new_origins, new_directions, bands_idx, ray_data


#        # Compute interaction with InterfaceManager
#        if self.max_interactions == 0:
#            self.interface.compute(self.acoustic_rays, hit_points, normals, absorption, reflection, refraction, scattering, hit_objects, bands_idx, source_medium)
#        else:
#            self.interface.compute(self.acoustic_rays, hit_points, normals, absorption, reflection, refraction, scattering, hit_objects, bands_idx)
#
#        # Generate new rays based on material properties
#        next_positions, next_directions = self._generate_scattered_rays(hit_points, normals, absorption, reflection, refraction, scattering, hit_objects)
#
#        # Record computed data to AcoustiRay
#
#        # Convert lists to arrays
#        if next_positions:
#            next_positions = np.vstack(next_positions)
#            next_directions = np.vstack(next_directions)
#        else:
#            next_positions = np.array([])
#            next_directions = np.array([])
        


    @staticmethod
#    @nb.njit(fastmath=True)
    def _random_hemisphere_directions(normals: np.ndarray) -> np.ndarray:
        """
        Generate random directions on hemispheres oriented along the normals.
        
        Args:
            normals: Surface normals
            n_samples: Number of samples to generate
            
        Returns:
            Array of sampled directions
        """
        n_samples = max(normals.shape[0], 1)
        directions = np.random.uniform(-1,1,(n_samples,3))
        directions /= np.linalg.norm(directions)

        while not np.all(directions[:, 0]**2 + directions[:, 1]**2 + directions[:, 2]**2 < 1):
            directions = np.random.uniform(-1,1,(n_samples,3))

            # Project onto hemisphere oriented along normals
            directions /= np.linalg.norm(directions)
            
        # Flip if pointing away from normals
        if not np.any(np.sum(directions * normals, axis=1) < 0):
            mask = np.sum(directions * normals, axis=1) < 0
            directions[mask] = -directions[mask]
        
        return directions

    def _compute_reflections(self, directions: np.ndarray, normals: np.ndarray):
        """
        Compute reflected angles for rays intersecting with surfaces.

        Parameters:
        -----------
        directions : numpy.ndarray
            Direction vectors of rays, shape (n_rays, 3)
        normals : numpy.ndarray
            Surface normals at intersection points, shape (n_rays, 3)

        Returns:
        --------
        incident_angles : numpy.ndarray
            Incident angles in radians, shape (n_rays,)
        reflected_directions : numpy.ndarray
            Reflected directions, shape (n_rays, 3)
        """
        # Compute reflected directions
        dot = np.sum(directions * normals, axis=1)
        incident_angles = np.arccos(-dot)
        reflected_directions = directions - 2 * dot[:, np.newaxis] * normals

        return incident_angles, reflected_directions
