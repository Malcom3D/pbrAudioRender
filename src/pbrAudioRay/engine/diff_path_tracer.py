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
from typing import Tuple, Optional, List, Any, Dict
from dataclasses import dataclass, field

@dataclass
class DiffPathTracer:
    """
    Implements differentiable path tracing for acoustic rendering.
    Based on: https://pub.dega-akustik.de/DAGA_2024/files/upload/paper/489.pdf
    """
    acoustic_scene: Any  # AcousticScene
    acoustic_rays: Any   # AcousticRay
    
    def __post_init__(self):
        # Initialize counters
        self.ray_count = 0
        self.max_depth = 10
        
    def compute(self, hits: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process ray hits and generate new rays for next bounce.
        
        Args:
            hits: Dictionary containing ray intersection results from Embree
            
        Returns:
            Tuple of (next_source_positions, next_directions)
        """
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

        geom_ids = hits["geomID"]
        prim_ids = hits["primID"]
        u_coords = hits["u"]
        v_coords = hits["v"]

        # Filter rays that hit something
        valid_hits = geom_ids >= 0
        if not np.any(valid_hits):
            return np.array([]), np.array([])
        
        # Get valid hit data
        valid_geom_ids = geom_ids[valid_hits]
        valid_prim_ids = prim_ids[valid_hits]
        valid_u = u_coords[valid_hits]
        valid_v = v_coords[valid_hits]
        
        # Process hits to determine next ray origins and directions
        next_positions, next_directions = self._process_hits(valid_geom_ids, valid_prim_ids, valid_u, valid_v)

        return next_positions, next_directions
    
    def _process_hits(self, geom_ids: np.ndarray, prim_ids: np.ndarray, u_coords: np.ndarray, v_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process ray hits and generate new rays based on material properties.
        
        Args:
            geom_ids: Geometry IDs of hits
            prim_ids: Primitive IDs of hits
            u_coords: Barycentric u coordinates
            v_coords: Barycentric v coordinates
            
        Returns:
            Tuple of (next_positions, next_directions)
        """
        n_hits = len(geom_ids)
        
        # Initialize arrays for next ray data
        next_positions = []
        next_directions = []
        
        # Get scene data
        mesh_info = self.acoustic_scene.mesh_info
        scene_info = self.acoustic_scene.scene_info
        
        # Process each hit
        for i in range(n_hits):
            geom_id = geom_ids[i]
            prim_id = prim_ids[i]
            u = u_coords[i]
            v = v_coords[i]
            w = 1.0 - u - v
            
            # Get hit triangle vertices
            triangle = mesh_info[prim_id]
            v0, v1, v2 = triangle[0], triangle[1], triangle[2]
            
            # Compute hit point using barycentric coordinates
            hit_point = w * v0 + u * v1 + v * v2
            
            # Get object index from scene info
            obj_idx = scene_info[prim_id]
            
            # Handle different object types
            if obj_idx == -3:  # Output hit - store for IR and stop tracing
                self._store_output_hit(hit_point, prim_id)
                continue
            elif obj_idx == -2:  # Source hit - discard
                continue
            elif obj_idx == -1:  # Acoustic domain hit - lost
                continue
            else:  # Object hit - generate new rays
                # Get material properties
                absorption = self.acoustic_scene.absorption[prim_id]
                reflection = self.acoustic_scene.reflection[prim_id]
                refraction = self.acoustic_scene.refraction[prim_id]
                scattering = self.acoustic_scene.scattering[prim_id]
                
                # Compute triangle normal
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm_length = np.linalg.norm(normal)
                if norm_length > 1e-12:
                    normal = normal / norm_length
                
                # Generate new rays based on material properties
                new_pos, new_dirs = self._generate_scattered_rays(hit_point, normal, absorption, reflection, refraction, scattering, obj_idx)
                
                if new_pos is not None and new_dirs is not None:
                    next_positions.append(new_pos)
                    next_directions.append(new_dirs)
        
        # Convert lists to arrays
        if next_positions:
            next_positions = np.vstack(next_positions)
            next_directions = np.vstack(next_directions)
        else:
            next_positions = np.array([])
            next_directions = np.array([])
        
        return next_positions, next_directions
    
    def _generate_scattered_rays(self, hit_point: np.ndarray, normal: np.ndarray, absorption: np.ndarray, reflection: np.ndarray, refraction: np.ndarray, scattering: np.ndarray, obj_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate scattered rays based on material properties.
        
        Args:
            hit_point: Point of intersection
            normal: Surface normal at hit point
            absorption: Absorption coefficients
            reflection: Reflection coefficients
            refraction: Refraction coefficients
            scattering: Scattering coefficients
            obj_idx: Object index
            
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
    @nb.njit(fastmath=True)
    def _sample_hemisphere(normal: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Sample directions on a hemisphere oriented along the normal.
        
        Args:
            normal: Surface normal
            n_samples: Number of samples to generate
            
        Returns:
            Array of sampled directions
        """
        directions = np.zeros((n_samples, 3), dtype=np.float32)
        
        for i in range(n_samples):
            # Generate random point on unit sphere
            while True:
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1)
                z = np.random.uniform(-1, 1)
                
                if x*x + y*y + z*z < 1.0:
                    break
            
            # Project onto hemisphere oriented along normal
            dir_vec = np.array([x, y, z])
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            
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
