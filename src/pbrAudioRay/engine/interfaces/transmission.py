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

from pbrAudioCommon import np
import numba as nb
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ...core.entity_manager import EntityManager

@dataclass
class TransmissionInterface:
    entity_manager: EntityManager

    def compute(self, medium_objs: np.ndarray, hits_obj_idx: np.ndarray, material_properties: Any, inters: np.ndarray, primID_filtered: np.ndarray, normals: np.ndarray, ray_data: Any, absorbed_energies: np.ndarray, frame_idx: int):
        """
        Compute transmission of rays through objects.
        
        Args:
            medium_objs: Medium object indices for each ray
            hits_obj_idx: Hit object indices for each ray
            material_properties: Material properties for intersected faces
            primID_filtered: Primitive indices for intersected faces
            normals: Surface normals at intersection points
            ray_data: Current ray data (origins, directions, energies, etc.)
            absorbed_energies: Energy absorbed at the surface
            
        Returns:
            Dictionary with transmission ray data
        """
        config = self.entity_manager.get('config')
        objects = self.entity_manager.get('objects')

        # Compute max number of transmitted rays
#        n_rays = ray_data.origins.shape[0]
        n_rays = inters.shape[0]
        max_transmission = int(n_rays * config.interface.max_transmission)

        # Compute transmission energies and phases
        trans_coeffs = material_properties.transmission_coeffs[primID_filtered][:, ray_data.bands_idx]
        trans_phases = material_properties.transmission_phases[primID_filtered][:, ray_data.bands_idx]

        # Compute incident angles
        dot_projection = np.sum(ray_data.directions * normals, axis=1)
        incident_angles = np.arccos(-dot_projection)

        # Compute transmission origins
        transmission_origins = inters - (0.01 * normals)
        transmission_origins = transmission_origins.astype(np.float32)

        # Compute incident directions
        incident_directions = ray_data.origins - transmission_origins

        # Get sound speeds for Snell's law
        medium_speeds = self._get_medium_speeds(medium_objs, ray_data.bands_idx)

        # Find next medium for transmitted rays
        new_medium_speed = np.full((n_rays, 1), config.acoustic_domain.acoustic_shader.sound_speed, dtype=np.float32)
        for key in objects:
            mesh = objects[key].get_mesh(frame_idx)
            for obj_config in config.objects:
                if obj_config.idx == objects[key].obj_idx:
                    medium_mask = mesh.contains(transmission_origins)
                    if np.any(medium_mask):
                        new_medium_speed[medium_mask] = obj_config.acoustic_shader.sound_speed

        # Determine which rays transmit based on energy and angle
        transmission_mask = self._compute_transmission_mask(trans_coeffs, incident_angles, medium_speeds, new_medium_speed, max_transmission)

        if not np.any(transmission_mask):
            return self._empty_result()

        # Compute transmission directions using Snell's Law
        transmission_directions = self._compute_transmission_directions(ray_data.directions[transmission_mask], normals[transmission_mask], incident_angles[transmission_mask], medium_speeds[transmission_mask], new_medium_speed[transmission_mask])

        # Compute transmitted energies and phases
        transmitted_energies = ray_data.energies[transmission_mask] * trans_coeffs[transmission_mask].reshape(-1, 1)
        transmitted_phases = (ray_data.phases[transmission_mask] + trans_phases[transmission_mask].reshape(-1, 1)) % (2 * np.pi)

        # Apply energy conservation
        total_energy = np.sum(absorbed_energies[transmission_mask]) + np.sum(transmitted_energies)
        if total_energy > 0:
            scale = np.sum(ray_data.energies[transmission_mask]) / total_energy
            transmitted_energies *= scale
        
        return {
            'origins': transmission_origins[transmission_mask],
            'directions': transmission_directions,
            'normals': normals[transmission_mask],
            'energies': transmitted_energies,
            'phases': transmitted_phases,
            'delay': ray_data.delay[transmission_mask]
        }

    def _compute_transmission_mask(self, trans_coeffs: np.ndarray, incident_angles: np.ndarray, medium_speeds: np.ndarray, new_medium_speed: np.ndarray, max_transmission: int) -> np.ndarray:
        """Determine which rays should transmit based on physical criteria."""
        # Energy-based threshold
        energy_mask = trans_coeffs > 0.01
        
        # Angle-based mask (total internal reflection check)
        # Rays with very grazing angles are less likely to transmit
        angle_mask = np.abs(incident_angles) < np.pi / 3  # Within 60 degrees of normal
        
        # Speed ratio check for Snell's law validity
        speed_ratio = medium_speeds.flatten() / new_medium_speed.flatten()
        valid_refraction = np.abs(speed_ratio * np.sin(incident_angles)) <= 1.0
        
        # Combine masks
        transmission_mask = energy_mask & angle_mask & valid_refraction
        
        # Limit number of transmission rays for performance
        if np.count_nonzero(transmission_mask) > max_transmission:
            # Keep only the strongest transmission rays
            indices = np.where(transmission_mask)[0]
            energies = trans_coeffs[indices]
            keep_indices = indices[np.argsort(energies)[-max_transmission:]]
            transmission_mask = np.zeros_like(transmission_mask)
            transmission_mask[keep_indices] = True
        
        return transmission_mask

    def _get_medium_speeds(self, medium_objs: np.ndarray, bands_idx: int) -> np.ndarray:
        """Get sound speeds for each ray based on medium type."""
        config = self.entity_manager.get('config')
        
        # Default speed of sound in air
        default_speed = config.acoustic_domain.acoustic_shader.sound_speed
        
        # Get object-specific sound speeds
        speeds = np.full(medium_objs.shape[0], default_speed, dtype=np.float32)
        
        for obj_config in config.objects:
            if obj_config.acoustic_shader and hasattr(obj_config.acoustic_shader, 'sound_speed'):
                mask = medium_objs.flatten() == obj_config.idx
                if np.any(mask):
                    speeds[mask] = obj_config.acoustic_shader.sound_speed
        
        return speeds.reshape(-1, 1)

    def _compute_transmission_directions(self, incident_directions: np.ndarray, normals: np.ndarray, incident_angles: np.ndarray, medium_speeds: np.ndarray, new_medium_speed: np.ndarray) -> np.ndarray:
        """
        Compute transmission directions using Snell's Law with SIMD optimization.
        
        Snell's Law: v2 * sin(θ1) = v1 * sin(θ2)
        where n = c0 / c (refractive index)
        """
        n_rays = incident_directions.shape[0]
        transmission_directions = np.zeros_like(incident_directions)
        
        # Use Numba for SIMD acceleration
        self._compute_transmission_directions_numba(incident_directions, normals, incident_angles, medium_speeds, new_medium_speed, transmission_directions)
        
        return transmission_directions

    @staticmethod
#    @nb.jit(nopython=True, parallel=True, fastmath=True)
    def _compute_transmission_directions_numba(incident_directions: np.ndarray, normals: np.ndarray, incident_angles: np.ndarray, medium_speeds: np.ndarray, new_medium_speed: np.ndarray, transmission_directions: np.ndarray):
        """
        Numba-accelerated computation of transmission directions using Snell's Law.
        """
        n_rays = incident_directions.shape[0]
        c0 = new_medium_speed
        
        for i in nb.prange(n_rays):
            # Normalize inputs
            incident_dir = incident_directions[i] / np.linalg.norm(incident_directions[i])
            normal = normals[i] / np.linalg.norm(normals[i])
            
            theta_i = incident_angles[i]
            
            v1 = medium_speeds[i, 0] # First medium
            v2 = c0[i, 0]  # Second medium
            
            # Snell's Law: v2 * sin(θ1) = v1 * sin(θ2)
            sin_theta_i = np.sin(theta_i)
            sin_theta_t = (v2 / v1) * sin_theta_i
            
            # Check for total internal reflection
            if sin_theta_t > 1.0:
                # Total internal reflection - transmit in same direction
                transmission_directions[i] = incident_dir
            else:
                theta_t = np.arcsin(sin_theta_t)
                
                # Decompose incident direction into normal and tangential components
                n_dot_i = np.dot(normal, incident_dir)
                incident_normal_component = n_dot_i * normal
                incident_tangent_component = incident_dir - incident_normal_component
                
                # Normalize tangent component
                tangent_norm = np.linalg.norm(incident_tangent_component)
                if tangent_norm > 1e-10:
                    incident_tangent_unit = incident_tangent_component / tangent_norm
                else:
                    incident_tangent_unit = np.zeros(3)
                
                # Compute transmission direction
                # The ray continues in the same tangential direction but refracts
                transmission_direction = (np.cos(theta_t) * normal - np.sin(theta_t) * incident_tangent_unit)
                
                # Normalize
                transmission_direction = transmission_direction / np.linalg.norm(transmission_direction)
                transmission_directions[i] = transmission_direction


    def _empty_result(self) -> Dict[str, np.ndarray]:
        """Return empty result when no transmission occurs."""
        return {
            'origins': np.zeros((0, 3), dtype=np.float32),
            'directions': np.zeros((0, 3), dtype=np.float32),
            'normals': np.zeros((0, 3), dtype=np.float32),
            'energies': np.zeros((0, 1), dtype=np.float32),
            'phases': np.zeros((0, 1), dtype=np.float32),
            'delay': np.zeros((0, 1), dtype=np.float32)
        }
