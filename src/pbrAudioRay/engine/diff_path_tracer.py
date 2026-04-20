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

    def compute(self, hits: Any, source_pos: np.ndarray, output_pos: np.ndarray):
        mesh_info = self.acoustic_scene.mesh_info
        scene_info = self.acoustic_scene.scene_info

        ray_inter = hits["geomID"] >= 0
        primID = hits["primID"][ray_inter]
        u = hits["u"][ray_inter]
        v = hits["v"][ray_inter]
        w = 1 - u - v
        next_source_pos = (np.vstack(w) * mesh_info[primID][:, 0, :] + np.vstack(u) * mesh_info[primID][:, 1, :] + np.vstack(v) * mesh_info[primID][:, 2, :])
        print('DiffPathTracer: ', len(source_pos), len(next_source_pos))
        dists = next_source_pos - source_pos
        delay = dists / 343.4
        output, hits_obj_idx = self._find_output_and_obj_idx(scene_info[primID])

        n_dirs = next_source_pos.shape[0]
        next_directions = self._generate_isotropic_directions(source_pos, output_pos, n_dirs)

        return next_source_pos, next_directions

    @staticmethod
    @nb.njit(fastmath=True)
    def _find_output_and_obj_idx(raw_obj_idx):
        """
        Optimized version using boolean masks. 
        """
        arr = np.asarray(raw_obj_idx, dtype=np.int32)

        # Create boolean masks (SIMD operations)
        mask_minus_three = (arr == -3)
        mask_non_negative = (arr >= 0)

        # Get indices from masks
        indices_output = np.flatnonzero(mask_minus_three)
        indices_obj_idx = np.flatnonzero(mask_non_negative)

        return indices_output, indices_obj_idx

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
