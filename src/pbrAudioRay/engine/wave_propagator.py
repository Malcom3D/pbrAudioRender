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
from numba import prange
from dask import delayed
from typing import List, Tuple
from dataclasses import dataclass

from ..core.entity_manager import EntityManager

from ..engine.ray_tracer import RayTracer
from ..engine.diff_path_tracer import DiffPathTracer

from ..lib.embree_scene import EmbreeScene
from ..lib.acoustic_ray import AcousticRay
from ..lib.simd_math import generate_all_directions_batch

@dataclass
class WavePropagator:
    """Optimized wave propagator using SIMD and parallel processing"""
    entity_manager: EntityManager
    combo: Tuple[int, int]
    
    def __post_init__(self):
        self.source_idx, self.output_idx = self.combo
        self.config = self.entity_manager.get('config')

        # Get frequency bands
        self.freq_bands = self.entity_manager.get('frequency_bands').get_bands()
        
    @delayed
    def compute(self, frame_idx):
        """Compute impulse response for a single frame"""
        # Get scene data for this frame
        embree_scene = EmbreeScene(self.entity_manager, self.combo, frame_idx)
        scene = embree_scene.scene
        acoustic_scene = embree_scene.acoustic_scene

        # Generate initial rays data structure
        n_rays = self.config.system.number_of_rays
        n_bands = len(self.freq_bands)
        max_interactions = self.config.wave_propagation.max_interactions

        # Init AcousticRay storage
        acoustic_rays = AcousticRay(n_rays, n_bands, max_interactions)

        # Init RayTracer engine
        self.ray_tracer = RayTracer(scene)

        # Init DiffPathTracer engine
        self.diff_path_tracer = DiffPathTracer(acoustic_scene, acoustic_rays)

        # compute first sources and directions
        source_pos = embree_scene.src_pos
        output_pos = embree_scene.out_pos

        # Diffuse source
        for source in self.config.sources:
            if source.idx == self.source_idx:
                if source.type == 'SPERE' and source.size > 0:
                    source_size = source.size
                    n_points = int(np.random.uniform(1, 10, size=1))
                    source_pos = self._source_points(n_points, source_pos, source_size)

#        directions = self._generate_initial_directions(n_rays, source_pos, output_pos)

        n_src = source_pos.shape[0]
        source_ndim = int(n_rays / n_src)
        source_pos = np.full((source_ndim,3), [source_pos.tolist()], dtype=np.float32)

        n_dirs = source_ndim * n_src
        directions = self._generate_isotropic_directions(source_pos, output_pos, n_dirs)
        directions = np.array(directions, dtype=np.float32)

        self.compute_loop(source_pos, directions)

    def compute_loop(self, source_pos: np.ndarray, output_pos: np.ndarray, directions: np.ndarray):
        hits = self.ray_tracer.compute(source_pos, directions)
        next_source_pos, next_directions = self.diff_path_tracer.compute(hits, source_pos, output_pos)

        if not next_source_pos and not next_directions:
            self.compute_loop(next_source_pos, next_directions)

        print('WavePropagator: compute_loop end')

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


    @staticmethod
    @nb.njit(fastmath=True)
    def _source_points(n_points: int, source_center: np.ndarray, source_size: float) -> np.ndarray:
        """
        Generate random points uniformly distributed inside a sphere using Marsaglia's method.
        More efficient than rejection sampling.
        """
        points = np.zeros((n_points, 3))
        cx, cy, cz = source_center[0], source_center[1], source_center[2]

        for i in range(n_points):
            # Marsaglia's method for uniform distribution in sphere
            while True:
                # Generate random point on unit disk
                u = 2.0 * np.random.random() - 1.0
                v = 2.0 * np.random.random() - 1.0
                s = u*u + v*v

                if s < 1.0:
                    # Generate random radius with cubic root for uniform volume distribution
                    r = source_size * np.cbrt(np.random.random())

                    # Calculate coordinates
                    sqrt_term = np.sqrt(1.0 - s)
                    x = 2.0 * u * sqrt_term
                    y = 2.0 * v * sqrt_term
                    z = 1.0 - 2.0 * s

                    # Scale and translate
                    points[i, 0] = cx + r * x
                    points[i, 1] = cy + r * y
                    points[i, 2] = cz + r * z
                    break
        return points

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
