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
from typing import Tuple, Optional, List, Any
from dataclasses import dataclass, field

from ..lib.simd_math import (
    reflect_batch, refract_batch, normalize_batch,
    compute_fresnel_coeffs_batch, importance_resample_batch
)

@dataclass
class DiffPathTracer:
    """
    Implements differentiable path tracing for acoustic rendering.
    Based on: https://pub.dega-akustik.de/DAGA_2024/files/upload/paper/489.pdf
    """
    acoustic_scene: Any  # AcousticScene
    acoustic_rays: Any   # AcousticRay
    
    def __post_init__(self):
        # Configuration parameters
        self.max_bounces = 10
        self.min_energy = 1e-6
        self.importance_sampling = True
        
    def compute(self, hits: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process ray hits and generate next ray directions using differentiable path tracing.
        
        Args:
            hits: Dictionary containing ray intersection results from Embree
            
        Returns:
            Tuple of (next_source_positions, next_directions) for continued tracing
        """
        # Extract hit information
        mesh_info = self.acoustic_scene.mesh_info
        scene_info = self.acoustic_scene.scene_info
        
        # Get intersection data
        ray_inter = hits["geomID"] >= 0
        if not np.any(ray_inter):
            return np.array([]), np.array([])
        
        primID = hits["primID"][ray_inter]
        u = hits["u"][ray_inter]
        v = hits["v"][ray_inter]
        w = 1 - u - v
        
        # Compute hit coordinates
        hit_triangles = mesh_info[primID]
        raw_source_pos = (
            w[:, np.newaxis] * hit_triangles[:, 0, :] +
            u[:, np.newaxis] * hit_triangles[:, 1, :] + 
            v[:, np.newaxis] * hit_triangles[:, 2, :]
        )
        
        # Get object indices for hit triangles
        hit_object_indices = scene_info[primID]
        
        # Process hits for differentiable path tracing
        next_positions, next_directions = self._process_hits_differentiable(
            raw_source_pos, hit_object_indices, primID, u, v, w
        )
        
        return next_positions, next_directions
    
    def _process_hits_differentiable(self, hit_points: np.ndarray, object_indices: np.ndarray,
                                   primID: np.ndarray, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process hits using differentiable path tracing with importance sampling.
        """
        n_hits = hit_points.shape[0]
        
        # Get material properties at hit points
        material_props = self._get_material_properties(object_indices, primID)
        
        # Compute surface normals at hit points
        normals = self._compute_surface_normals(primID, u, v)
        
        # Determine interaction types based on material properties and incidence angle
        interaction_types = self._determine_interaction_types(
            self.acoustic_rays.directions[:n_hits], normals, material_props
        )
        
        # Apply interactions and generate new directions
        new_directions = self._apply_interactions(
            self.acoustic_rays.directions[:n_hits], normals, material_props, interaction_types
        )
        
        # Update ray states with interaction information
        self._update_ray_states(hit_points, normals, material_props, interaction_types)
        
        # Apply importance sampling if enabled
        if self.importance_sampling:
            indices = importance_resample_batch(
                self.acoustic_rays.energy[:n_hits],
                self.acoustic_rays.gradients[:n_hits],
                n_hits  # Keep same number of rays
            )
            hit_points = hit_points[indices]
            new_directions = new_directions[indices]
        
        return hit_points, new_directions
    
    def _get_material_properties(self, object_indices: np.ndarray, primID: np.ndarray) -> Dict:
        """Extract material properties for hit triangles."""
        n_hits = object_indices.shape[0]
        n_bands = self.acoustic_scene.absorption.shape[2]
        
        # Initialize material properties array
        material_props = {
            'absorption': np.zeros((n_hits, n_bands), dtype=np.float32),
            'reflection': np.zeros((n_hits, n_bands), dtype=np.float32),
            'refraction': np.zeros((n_hits, n_bands), dtype=np.float32),
            'scattering': np.zeros((n_hits, n_bands), dtype=np.float32),
            'diffraction': np.zeros((n_hits, n_bands), dtype=np.float32),
            'sound_speed': np.zeros(n_hits, dtype=np.float32),
            'density': np.zeros(n_hits, dtype=np.float32)
        }
        
        # Fill material properties based on object indices
        for i in range(n_hits):
            obj_idx = object_indices[i]
            tri_idx = primID[i]
            
            if obj_idx >= 0:  # Acoustic object
                material_props['absorption'][i] = self.acoustic_scene.absorption[tri_idx, 0, :]
                material_props['reflection'][i] = self.acoustic_scene.reflection[tri_idx, 0, :]
                material_props['refraction'][i] = self.acoustic_scene.refraction[tri_idx, 0, :]
                material_props['scattering'][i] = self.acoustic_scene.scattering[tri_idx, 0, :]
                material_props['sound_speed'][i] = self.acoustic_scene.sound_speed[tri_idx]
                material_props['density'][i] = self.acoustic_scene.density[tri_idx]
            else:  # Domain, source, or output
                material_props['sound_speed'][i] = 343.0  # Default air speed
                material_props['density'][i] = 1.225  # Default air density
        
        return material_props
    
    def _compute_surface_normals(self, primID: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute interpolated surface normals at hit points."""
        n_hits = primID.shape[0]
        normals = np.zeros((n_hits, 3), dtype=np.float32)
        
        # Get triangle vertices
        triangles = self.acoustic_scene.mesh_info[primID]
        
        # Compute face normals (cross product of edges)
        for i in range(n_hits):
            v0, v1, v2 = triangles[i]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normals[i] = normal / norm
        
        return normals
    
    def _determine_interaction_types(self, incident_dirs: np.ndarray, normals: np.ndarray,
                                   material_props: Dict) -> np.ndarray:
        """
        Determine interaction type for each hit based on material properties and incidence.
        
        Returns array of interaction types:
        0: absorption (ray terminated)
        1: reflection
        2: refraction
        3: scattering
        4: diffraction
        """
        n_hits = incident_dirs.shape[0]
        interaction_types = np.zeros(n_hits, dtype=np.int32)
        
        # Compute cosine of incidence angle
        cos_theta = np.sum(incident_dirs * normals, axis=1)
        
        for i in range(n_hits):
            # Get material coefficients for this hit
            absorption = np.mean(material_props['absorption'][i])
            reflection = np.mean(material_props['reflection'][i])
            refraction = np.mean(material_props['refraction'][i])
            scattering = np.mean(material_props['scattering'][i])
            
            # Determine interaction type probabilistically
            total = absorption + reflection + refraction + scattering
            if total > 0:
                r = np.random.random()
                cum_prob = 0.0
                
                # Absorption
                cum_prob += absorption / total
                if r < cum_prob:
                    interaction_types[i] = 0
                    continue
                
                # Reflection
                cum_prob += reflection / total
                if r < cum_prob:
                    interaction_types[i] = 1
                    continue
                
                # Refraction
                cum_prob += refraction / total
                if r < cum_prob:
                    interaction_types[i] = 2
                    continue
                
                # Scattering
                interaction_types[i] = 3
            else:
                # Default to reflection if no coefficients specified
                interaction_types[i] = 1
        
        return interaction_types
    
    def _apply_interactions(self, incident_dirs: np.ndarray, normals: np.ndarray,
                          material_props: Dict, interaction_types: np.ndarray) -> np np.ndarray:
        """Apply interactions to generate new ray directions."""
        n_hits = incident_dirs.shape[0]
        new_directions = np.zeros_like(incident_dirs)
        
        for i in range(n_hits):
            if interaction_types[i] == 0:  # Absorption
                # Ray terminated, no new direction
                new_directions[i] = np.zeros(3)
                continue
                
            elif interaction_types[i] == 1:  # Reflection
                # Specular reflection
                new_directions[i] = reflect_batch(
                    incident_dirs[i:i+1], normals[i:i+1]
                )[0]
                
                
            elif interaction_types[i] == 2:  # Refraction
                # Snell's law refraction
                # For simplicity, assume air-to-material or material-to-air
                n1 = 1.0  # Air refractive index (approximate)
                n2 = material_props['density'][i] / 1.225  # Relative refractive index
                
                new_directions[i] = refract_batch(
                    incident_dirs[i:i+1], normals[i:i+1], n1, n2
                )[0]
                
            elif interaction_types[i] == 3:  # Scattering
                # Lambertian scattering (diffuse reflection)
                # Generate random direction on hemisphere
                theta = 2.0 * np.pi * np.random.random()
                phi = np.arccos(np.sqrt(np.random.random()))
                
                # Local coordinate system
                if np.abs(normals[i, 2]) < 0.999:
                    tangent = np.array([-normals[i, 1], normals[i, 0], 0])
                else:
                    tangent = np.array([0, -normals[i, 2], normals[i, 1]])
                tangent = tangent / np.linalg.norm(tangent)
                bitangent = np.cross(normals[i], tangent)
                
                # Convert to world coordinates
                scattered_dir = (
                    np.sin(phi) * np.cos(theta) * tangent +
                    np.sin(phi) * np.sin(theta) * bitangent +
                    np.cos(phi) * normals[i]
                )
                new_directions[i] = scattered_dir / np.linalg.norm(scattered_dir)
        
        return normalize_batch(new_directions)
    
    def _update_ray_states(self, hit_points: np.ndarray, normals: np.ndarray,
                          material_props: Dict, interaction_types: np.ndarray):
        """Update ray states with interaction information for differentiable tracing."""
        n_hits = hit_points.shape[0]
        n_bands = self.acoustic_rays.n_freq_bands
        
        for i in range(n_hits):
            if not self.acoustic_rays.active[i]:
                continue
                
            # Create interaction coefficients array
            coeffs = np.zeros((n_bands, 5), dtype=np.float32)
            for k in range(n_bands):
                coeffs[k, 0] = material_props['absorption'][i, k]  # absorption
                coeffs[k, 1] = material_props['reflection'][i, k]  # reflection
                coeffs[k, 2] = material_props['refraction'][i, k]  # refraction
                coeffs[k, 3] = material_props['scattering'][i, k]  # scattering
                coeffs[k, 4] = 0.0  # diffraction (not implemented yet)
            
            # Record interaction
            self.acoustic_rays.add_interaction(
                i, interaction_types[i], coeffs, normals[i], hit_points[i]
            )
            
            # Update ray depth and check termination
            self.acoustic_rays.depth[i] += 1
            if (self.acoustic_rays.depth[i] >= self.max_bounces or
                np.mean.mean(np.abs(self.acoustic_rays.energy[i])) < self.min_energy):
                self.acoustic_rays.active[i] = False
    
    def _generate_isotropic_directions(self, src: np.ndarray, dst: np.ndarray, 
                                     n_directions: int = 100, seed: int = None) -> List[np.ndarray]:
        """
        Generate random directions with isotropic probability distribution in 4π sr.
        (Keep your existing implementation)
        """
        if seed is not None:
            np.random.seed(seed)

        # Direct direction
        direct_vec = dst - src
        vec_norm = np.linalg.norm(direct_vec)
        if vec_norm < 1e-12:
            raise ValueError("Source and destination are coincident")
        direct_dir = direct_vec_vec / vec_norm

        # Generate isotropic directions
        isotropic_dirs = []

        for _ in range(n_directions):
            # Marsaglia method (1972) for uniform distribution on sphere
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
        return isotropic_dirs
