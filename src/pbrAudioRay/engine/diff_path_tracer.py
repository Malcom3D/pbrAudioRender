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

from .interface import InterfaceManager

@dataclass
class DiffPathTracer:
    """
    Implements differentiable path tracing for acoustic rendering.
    Based on: https://pub.dega-akustik.de/DAGA_2024/files/upload/paper/489.pdf
    """
    acoustic_scene: Any  # AcousticScene
    
    def __post_init__(self):
        # Initialize counters
        self.ray_count = 0
        self.current_interactions = 0
        self.max_interactions = self.acoustic_rays.max_interactions
        self.interface = InterfaceManager(self.acoustic_scene.freq_bands)

    @delayed
    def compute(self, hits: Dict, bands_idx: int, ray_data: Any) -> Tuple[np.ndarray, np.ndarray, int, Any]:
        """
        Process ray hits and generate new rays for next bounce.
        
        Args:
            hits: Dictionary containing ray intersection results from Embree
            
        Returns:
            Tuple of (next_source_positions, next_directions)
        """

        if self.current_interactions == self.max_interactions -1:
            next_positions = np.array([])
            next_directions = np.array([])

        # Get scene data
        mesh_info = self.acoustic_scene.mesh_info
        scene_info = self.acoustic_scene.scene_info

        sound_speed = self.acoustic_scene.sound_speed
        density = self.acoustic_scene.density
        absorption = self.acoustic_scene.absorption
        refraction = self.acoustic_scene.refraction
        reflection = self.acoustic_scene.reflection
        scattering = self.acoustic_scene.scattering

        source_pos = self.acoustic_scene.aso_pos[0]
        output_pos = self.acoustic_scene.aso_pos[1]

        source_medium = self.acoustic_scene.aso_medium[0]
        output_medium = self.acoustic_scene.aso_medium[1]

        geom_ids = hits["geomID"]
        prim_ids = hits["primID"]

        # Filter rays that hit something
        valid_hits = geom_ids >= 0
        if not np.any(valid_hits):
            return np.array([]), np.array([])
        
        u = hits["u"][valid_hits]
        v = hits["v"][valid_hits]
        w = 1 - u - v

        # Get valid hit data
        valid_geom_ids = geom_ids[valid_hits]
        valid_prim_ids = prim_ids[valid_hits]
        valid_hit_coords = hit_coords[valid_hits]
        
        n_hits = len(geom_ids)
        
        # Initialize arrays for next ray data
        next_positions = []
        next_directions = []
        
        triangle = mesh_info[prim_id]
        a = triangle[:, 0, :]
        b = triangle[:, 1, :]
        c = triangle[:, 2, :]

        # Compute hit point using barycentric coordinates
        hit_points = (np.vstack(w) * a + np.vstack(u) * b + np.vstack(v) * c)

        # Compute 
        # Get object index from scene info
        hit_obj_idx = scene_info[prim_id]
        output_mask = (hit_obj_idx == -3)
        intersect_mask = (hit_obj_idx >= 0)
        hit_output = hit_obj_idx[output_mask]
        hit_objects = hit_obj_idx[intersect_mask]
        self._store_output_hit(hit_point, prim_id)
 
        # Get material properties
        absorption = absorption[intersect_mask][:,:,bands_idx]
        reflection = reflection[intersect_mask][:,:,bands_idx]
        refraction = refraction[intersect_mask][:,:,bands_idx]
        scattering = scattering[intersect_mask][:,:,bands_idx]

        # Compute triangle normal using np.cross with broadcasting
        normals = np.cross(b-a, c-a)

        # Normalize (avoid division by zero)
        normals = np.linalg.norm(normals, axis=1, keepdims=True)

        # Add small epsilon to avoid division by zero
        normals = np.maximum(normals, 1e-12)

        # Compute interaction with InterfaceManager
        if self.max_interactions == 0:
            self.interface.compute(self.acoustic_rays, hit_points, normals, absorption, reflection, refraction, scattering, hit_objects, bands_idx, source_medium)
        else:
            self.interface.compute(self.acoustic_rays, hit_points, normals, absorption, reflection, refraction, scattering, hit_objects, bands_idx)

        # Generate new rays based on material properties
        next_positions, next_directions = self._generate_scattered_rays(hit_points, normals, absorption, reflection, refraction, scattering, hit_objects)

        # Record computed data to AcoustiRay

        # Convert lists to arrays
        if next_positions:
            next_positions = np.vstack(next_positions)
            next_directions = np.vstack(next_directions)
        else:
            next_positions = np.array([])
            next_directions = np.array([])
        
        self.current_interactions += 1

        return next_source_pos, next_directions, bands_idx, ray_data

    def _generate_scattered_rays(self, hit_points: np.ndarray, normals: np.ndarray, absorption: np.ndarray, reflection: np.ndarray, refraction: np.ndarray, scattering: np.ndarray, obj_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate scattered rays based on material properties.
        
        Args:
            hit_points: Point of intersection
            normals: Surface normals at hit points
            absorption: Absorption coefficients
            reflection: Reflection coefficients
            refraction: Refraction coefficients
            scattering: Scattering coefficients
            hit_objects: Objects index
            
        Returns:
            Tuple of (positions, directions) for new rays
        """
        # For now, implement simple reflection
        # In a complete implementation, you would:
        # 1. Sample reflection direction based on BRDF
        # 2. Sample refraction direction using Snell's law
        # 3. Sample scattering direction
        # 4. Apply energy conservation
        
        # Simple Lambertian reflection for demonstration
        n_rays = 4  # Generate  4 new rays per hit
        positions = np.tile(hit_point, (n_rays, 1))
        
        # Generate random directions on hemisphere
        directions = self._sample_hemisphere(normal, n_rays)

        # Apply material coefficients to ray energy
        self._apply_material_coefficients(directions, absorption, reflection, refraction, scattering, obj_idx)
        
        return positions, directions
    
    @staticmethod
#    @nb.njit(fastmath=True)
    def _sample_hemisphere(normal: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Sample directions on a hemisphere oriented along the normal.
        
        Args:
            normal: Surface normal
            n_samples: Number of samples to generate
            
        Returns:
            Array of sampled directions
        """
        directions = np.random.uniform(-1,1,(n_samples,3))

        while not np.all(directions[:, 0]**2 + directions[:, 1]**2 + directions[:, 2]**2 < 1):
            directions = np.random.uniform(-1,1,(n_samples,3))

            # Project onto hemisphere oriented along normal
            directions /= np.linalg.norm(directions)
            
            # Flip if pointing away from normal
            if np.dot(dir_vec, normal) < 0:
                dir_vec = -dir_vec
            
            directions[i] = dir_vec
        
        return directions

    def _apply_material_coefficients(self, directions: np.ndarray, absorption: np.ndarray, reflection: np.ndarray, refraction: np.ndarray, scattering: np.ndarray, obj_idx: int):
        """
        Apply material coefficients to ray directions and energies.
        
        Args:
            directions: Ray directions
            absorption: Absorption coefficients
            reflection: Reflection coefficients
            refraction: Refraction coefficients
            scattering: Scattering coefficients
            obj_idx: Object index
        """
        # This would update the acoustic_rays data structure
        # For now, just a placeholder
        pass
    
    def _store_output_hit(self, hit_point: np.ndarray, prim_id: int):
        """
        Store hit information for impulse response calculation.
        
        Args:
            hit_point: Point where ray hit output
            prim_id: Primitive ID of hit
        """
        # Store hit information in acoustic_rays
        # This would include:
        # - Path length
        # - Energy at each frequency band
        # - Phase information
        # - Interaction history
        
        # For now, just increment counter
        self.ray_count += 1

    def _compute_incident_angles(ray_directions, intersection_points, normals):
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
        normals_normalized = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
        # Compute the dot product between ray direction and normal
        # Note: We use the negative of ray direction since incident angle is measured
        # between the incoming ray and the surface normal
        dot_products = np.sum(-ray_directions_normalized * normals_normalized, axis=1)
    
        # Clamp dot products to [-1, 1] to avoid numerical issues
        dot_products = np.clip(dot_products, -1.0, 1.0)
    
        # Compute incident angles (arccos of dot product)
        incident_angles = np.arccos(dot_products)
    
        return incident_angles
