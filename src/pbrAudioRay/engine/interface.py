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
from ..lib.ray_data import RayData

#from .interfaces import AbsorptionInterface, ReflectionInterface, RefractionInterface, ScatteringInterface, DiffractionInterface
from .interfaces.absorption import AbsorptionInterface
from .interfaces.reflection import ReflectionInterface
from .interfaces.scattering import ScatteringInterface

@dataclass
class InterfaceManager:
    """
    Handle rays interaction with objects boundaries
    """
    entity_manager: EntityManager
    acoustic_scene: Any  # AcousticScene
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.max_interactions = config.wave_propagation.max_interactions
        self.n_bands = len(self.entity_manager.get('frequency_bands').get_bands())

        self.absorption_interface = AbsorptionInterface(self.entity_manager, self.acoustic_scene)
        self.reflection_interface = ReflectionInterface(self.entity_manager)
        self.scattering_interface = ScatteringInterface(self.entity_manager)

        # Initialize counters
        self.current_interactions = 0

    @delayed
    def parallel_compute(self, hits: Dict[str, np.ndarray], ray_data: RayData) -> RayData:
        return self.compute(hits, ray_data)

    def compute(self, hits: Dict[str, np.ndarray], ray_data: RayData) -> RayData:
        """
        Process ray hits and deliver data to interface subclasses.
        
        Args:
            hits: Dictionary containing ray intersection results from Embree run
            
        Returns:
            Tuple of (next_source_positions, next_directions, bands_idx, ray_data)
        """
        config = self.entity_manager.get('config')
        enable_absorption = config.interface.enable_absorption
        enable_reflection = config.interface.enable_reflection
        enable_scattering = config.interface.enable_scattering

        if self.current_interactions == self.max_interactions:
            next_positions = np.array([])
            next_directions = np.array([])

        origins = ray_data.origins
        bands_idx = ray_data.bands_idx

        geom_ids = hits["geomID"] >= 0
        prim_ids = hits["primID"][geom_ids]
        if prim_ids.shape[0] == 0:
            return None

        u = hits["u"][prim_ids]
        v = hits["v"][prim_ids]
        w = 1 - u - v

        triangle = self.acoustic_scene.get_mesh_info()
        a = triangle[prim_ids][:, 0, :]
        b = triangle[prim_ids][:, 1, :]
        c = triangle[prim_ids][:, 2, :]
        
        hits_obj_idx = self.acoustic_scene.scene_info[prim_ids]
        intersect_mask = hits_obj_idx >= 0

        # Compute hit point using barycentric coordinates
        hit_points = (np.vstack(w) * a + np.vstack(u) * b + np.vstack(v) * c)

        # Compute triangle normals using np.cross with broadcasting
        normals = np.cross(b-a, c-a)

        # Normalize (avoid division by zero)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        # Get materials properties
        abs_coeffs, abs_phases = self.acoustic_scene.get_absorption(mask=prim_ids, bands_idx=bands_idx)
        refl_coeffs, refl_phases = self.acoustic_scene.get_reflection(mask=prim_ids, bands_idx=bands_idx)
        refr_coeffs, refr_phases = self.acoustic_scene.get_refraction(mask=prim_ids, bands_idx=bands_idx)
        scat_coeffs, scat_phases = self.acoustic_scene.get_scattering(mask=prim_ids, bands_idx=bands_idx)

        rays_energies = absorbed_energy = reflected_energy = scattered_energy = ray_data.energies
        rays_phases = absorbed_phases = reflected_phases = scattered_phases = ray_data.phases
        reflected_directions = scattered_directions = None
        incident_angles = 1

        if enable_absorption:
            rays_energies, rays_phases = self.absorption_interface.compute_attenuation(hit_points, bands_idx, ray_data)

            rays_energies = rays_energies[prim_ids]
            rays_energies = rays_energies[intersect_mask]
            rays_phases = rays_phases[prim_ids]
            rays_phases = rays_phases[intersect_mask]

        if enable_reflection:
            directions = ray_data.directions[prim_ids]
            reflected_energy, reflected_phases, incident_angles, reflected_directions = self.reflection_interface.compute(normals, directions, rays_energies, rays_phases, refl_coeffs, refl_phases, ray_data)

        if enable_absorption:
            absorbed_energy, absorbed_phases = self.absorption_interface.compute(rays_energies, rays_phases, incident_angles, abs_coeffs, abs_phases, ray_data)

        if enable_scattering:
            roughness_factor = self.acoustic_scene.get_roughness(mask=prim_ids)
            scattered_energy, scattered_phases, scattered_directions = self.scattering_interface.compute(rays_energies, rays_phases, normals, roughness_factor, scat_coeffs, scat_phases)

        absorbed_energy, scattered_energy, reflected_energy = self._check_energy_conservation(rays_energies, absorbed_energy, reflected_energy, scattered_energy)

        # Append hit_points for origins, directions, energies and phases
        if isinstance(reflected_directions, np.ndarray) and isinstance(scattered_directions, np.ndarray):
            appended_origins = np.append(hit_points, hit_points, axis=0).astype(np.float32)
            appended_directions = np.append(reflected_directions, scattered_directions, axis=0).astype(np.float32)
            appended_energies = np.append(reflected_energy, scattered_energy, axis=0).astype(np.float32)
            appended_phases = np.append(reflected_phases, scattered_phases, axis=0).astype(np.float32)
        elif isinstance(reflected_directions, np.ndarray) and not isinstance(scattered_directions, np.ndarray):
            appended_origins = hit_points
            appended_directions = reflected_directions
            appended_energies = reflected_energy
            appended_phases = reflected_phases
        elif not isinstance(reflected_directions, np.ndarray) and isinstance(scattered_directions, np.ndarray):
            appended_origins = hit_points
            appended_directions = scattered_directions
            appended_energies = scattered_energy
            appended_phases = scattered_phases
        
        # Filter direction, hit_points and phases on energy termination
        termination_energy = 1e-6 # config.termination.energy_threshold
        termination_mask = appended_energies > termination_energy
        new_energies = appended_energies[termination_mask].reshape(-1,1)
        new_phases = appended_phases[termination_mask].reshape(-1,1)
        new_directions = appended_directions[termination_mask.reshape(-1,)]
        new_origins = appended_origins[termination_mask.reshape(-1,)]

        # Create new RayData storage
        recursion_idx = ray_data.recursion_idx + 1
        new_ray_data = RayData(src_idx=ray_data.src_idx, out_idx=ray_data.out_idx, bands_idx=ray_data.bands_idx, recursion_idx=recursion_idx, origins=new_origins, directions=new_directions, energies=new_energies, phases=new_phases)

        self.current_interactions += 1
        return new_ray_data

    def _check_energy_conservation(self, rays_energies: np.ndarray, absorbed_energy: np.ndarray = None, reflected_energy: np.ndarray = None, scattered_energy: np.ndarray = None):
        # Energy conservation check
        total_out = rays_energies
        if isinstance(absorbed_energy, np.ndarray):
            total_out += absorbed_energy
        if isinstance(reflected_energy, np.ndarray):
            total_out += reflected_energy
        if isinstance(scattered_energy, np.ndarray):
            total_out += scattered_energy

        delta_energies = abs(total_out - rays_energies)
        delta_mask = delta_energies > 1e-10
        if not np.all(delta_mask):
            total_out[~delta_mask] = rays_energies[~delta_mask] + 1e-10
        # Normalize to ensure energy conservation
        scale = rays_energies / total_out
        if isinstance(absorbed_energy, np.ndarray):
            absorbed_energy *= scale
        if isinstance(reflected_energy, np.ndarray):
            reflected_energy *= scale
        if isinstance(scattered_energy, np.ndarray): 
            scattered_energy *= scale

        return absorbed_energy, reflected_energy, scattered_energy
