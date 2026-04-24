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
from ..lib.ray_data import RayData
from ..lib.simd_math import generate_all_directions_batch

@dataclass
class WavePropagator:
    """Wave propagator using SIMD and parallel processing"""
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
        self.recursion_idx = 0

        # Init RayTracer engine
        self.ray_tracer = RayTracer(scene)

        # Init DiffPathTracer engine
        self.diff_path_tracer = DiffPathTracer(self.entity_manager, acoustic_scene)

        # compute first sources and directions
        source_pos = acoustic_scene.aso_pos[0]
        output_pos = acoustic_scene.aso_pos[1]

        # Diffuse source
        for source in self.config.sources:
            if source.idx == self.source_idx:
                if source.type == 'SPERE' and source.size > 0:
                    source_size = source.size
                    n_points = int(np.random.uniform(1, 10, size=1))
                    source_pos = self._source_points(n_points, source_pos, source_size)

        n_src = source_pos.shape[0]
        source_ndim = int(n_rays / n_src)
        n_dirs = source_ndim * n_src
        print('source_pos: ', n_src, source_ndim, n_dirs, source_pos.shape, source_pos)
        source_pos = np.array([source_pos.tolist() for _ in range(source_ndim)]).reshape(n_dirs,3)
        print('source_pos: ', n_src, source_ndim, n_dirs, source_pos.shape)
        source_pos = np.full((n_dirs,3), [source_pos], dtype=np.float32)
        print('source_pos: ', n_src, source_ndim, n_dirs, source_pos.shape)

        directions = self._generate_isotropic_directions(n_dirs, source_pos, output_pos)
        directions = np.array(directions, dtype=np.float32)

        # First fast rays propagation without frequency bands
        hits = self.ray_tracer.compute(source_pos, directions)

        # Compute Paths for band_idx
        task_tracer = []
        for bands_idx in range(n_bands):
            # Init  RayData storage
            ray_data = RayData(self.source_idx, self.output_idx, bands_idx)
            ray_data.add_data(recursion_idx=self.recursion_idx, n_rays=source_pos.shape[0], origins=source_pos, directions=directions)
            _ = self.entity_manager.register('ray_datas', ray_data)
            task_tracer += [self.diff_path_tracer.compute(hits, bands_idx, ray_data)]
        tracer_results = compute(*task_tracer)

        for next_source_pos, next_directions, bands_idx, ray_data in tracer_results:
            if isinstance(next_source_pos, np.ndarray) and isinstance(next_directions, np.ndarray):
                if not next_source_pos.shape[0] == 0 and not next_directions.shape[0] == 0:
                    self.recursion_idx += 1
                    self.compute_loop(next_source_pos, next_directions, bands_idx, ray_data)

    def compute_loop(self, source_pos: np.ndarray, directions: np.ndarray, bands_idx: int):
        hits = self.ray_tracer.compute(source_pos, directions)
        next_source_pos, next_directions, bands_idx, ray_data = self.diff_path_tracer.compute(hits, bands_idx, ray_data)

        if isinstance(next_source_pos, np.ndarray) and isinstance(next_directions, np.ndarray):
            if not next_source_pos.shape[0] == 0 and not next_directions.shape[0] == 0:
                self.compute_loop(next_source_pos, next_directions, bands_idx)

        print('WavePropagator: compute_loop end')

    @staticmethod
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

    def _generate_isotropic_directions(self, n_dirs: int, src: np.ndarray, dst: np.ndarray, seed: int = 1) -> np.ndarray:
        """
        Generate n_dirs isotropic directions using vectorised rejection sampling,
        plus one direct direction toward dst.

        Parameters
        ----------
        n_dirs : int
            Number of isotropic directions to generate.
        src : np.ndarray
            Source position (3,).
        dst : np.ndarray
            Destination position (3,).
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        directions : np.ndarray
            Array of shape (n_dirs + 1, 3) with the direct direction last.
        """
        rng = np.random.default_rng(seed)

        # Direct direction
        direct_vec = dst - src
        norm = np.linalg.norm(direct_vec)
        if norm < 1e-12:
            direct_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            direct_dir = (direct_vec / norm).astype(np.float32)

        # Estimate required samples (acceptance probability π/4 ≈ 0.785)
        batch_factor = 1.5
        accepted_u1 = []
        accepted_u2 = []
        accepted_s = []
        needed = n_dirs

        while needed > 0:
            batch_size = int(needed * batch_factor)
            # Generate uniform numbers in [-1, 1]
            u1 = rng.uniform(-1.0, 1.0, size=batch_size).astype(np.float32)
            u2 = rng.uniform(-1.0, 1.0, size=batch_size).astype(np.float32)
            s = u1 * u1 + u2 * u2
            mask = s < 1.0
            # Keep accepted values
            accepted_u1.append(u1[mask])
            accepted_u2.append(u2[mask])
            accepted_s.append(s[mask])
            needed -= mask.sum()

        # Concatenate all batches
        u1_all = np.concatenate(accepted_u1)[:n_dirs]
        u2_all = np.concatenate(accepted_u2)[:n_dirs]
        s_all = np.concatenate(accepted_s)[:n_dirs]

        # Compute directions using Marsaglia's method
        sqrt_term = np.sqrt(1.0 - s_all)
        x = 2.0 * u1_all * sqrt_term
        y = 2.0 * u2_all * sqrt_term
        z = 1.0 - 2.0 * s_all

        # Stack into (n_dirs, 3) array
        isotropic = np.column_stack((x, y, z)).astype(np.float32)

        # Append direct direction
        return np.vstack((isotropic, direct_dir))
