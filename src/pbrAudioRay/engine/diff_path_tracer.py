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

@dataclass
class DiffPathTracer:
    """
    Implements differentiable path tracing for acoustic rendering
    Based on: https://pub.dega-akustik.de/DAGA_2024/files/upload/paper/489.pdf
    """
    acoustic_scene: Any # AcousticScene
    acoustic_rays: Any # AcousticRay

    def compute(self, hits: Any):
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

        ray_inter = hits["geomID"] >= 0
        primID = hits["primID"][ray_inter]
        u = hits["u"][ray_inter]
        v = hits["v"][ray_inter]
        w = 1 - u - v

        # compute raw hit coords and dists
        raw_source_pos = (np.vstack(w) * mesh_info[primID][:, 0, :] + np.vstack(u) * mesh_info[primID][:, 1, :] + np.vstack(v) * mesh_info[primID][:, 2, :])
        raw_dists = raw_source_pos - source_pos

        mask_non_negative = (scene_info[primID] >= 0)
        mask_from_ac = (scene_info[primID] >= -1)
        mask_output = (scene_info[primID] == -3)
 
        # filter raw hit coords and pos from output, source and acoustic domain
        output_ray = raw_source_pos[mask_output]
        next_source_pos = raw_source_pos[mask_non_negative]

        dists = raw_dists[mask_non_negative]
        hit_obj_idx = scene_info[mask_non_negative]

        obj_sound_speed = sound_speed[mask_from_ac]
        obj_density = density[mask_from_ac]
        obj_absorption = absorption[mask_from_ac]
        obj_refraction = refraction[mask_from_ac]
        obj_reflection = reflection[mask_from_ac]
        obj_scattering = scattering[mask_from_ac]

        delay = dists / obj_sound_speed

        n_dirs = next_source_pos.shape[0]
        next_directions = self._generate_isotropic_directions(next_source_pos, output_pos, n_dirs)

        return next_source_pos, next_directions

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

            direction = [x, y, z]
            isotropic_dirs.append(direction)
#        isotropic_dirs += [direct_dir.tolist()]
        return isotropic_dirs
