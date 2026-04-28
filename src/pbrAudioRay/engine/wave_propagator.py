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

import math
import numpy as np
import numba as nb
from numba import prange
from dask import delayed, compute
from typing import List, Tuple
from dataclasses import dataclass

from ..core.entity_manager import EntityManager

from ..engine.ray_tracer import RayTracer
from ..engine.diff_path_tracer import DiffPathTracer

from ..lib.functions import _load_pose
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
        print('Embreex: scene loading complete...', self.combo)
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

#        # compute first sources and directions
#        source_pos = acoustic_scene.aso_pos[0]
#        output_pos = acoustic_scene.aso_pos[1]

        # Load output positions
        for output_config in self.config.outputs:
            if output_config.idx == self.output_idx:
                output_positions, _ = _load_pose(output_config)
                if output_config.static:
                    output_pos = output_positions
                else:
                    output_pos = output_positions[frame_idx]

        # Load source positions
        for source_config in self.config.sources:
            if source_config.idx == self.source_idx:
                source_positions, _ = _load_pose(source_config)
                if source_config.static:
                    source_pos = source_positions
                else:
                    source_pos = source_positions[frame_idx]
                if source_config.type == 'SPHERE' and source_config.size > 0:
                    source_size = source_config.size
                    # Diffuse source
                    n_points = int(np.random.uniform(1, 10, size=1))
                    source_pos = self._source_points(n_points, source_pos, source_size)

        print('WavePropagator: sources: ', source_pos.shape)
        if source_pos.ndim == 1:
            source_pos = source_pos.reshape(1,-1)
        n_src = source_pos.shape[0]
        source_ndim = int(n_rays / n_src)
        n_dirs = source_ndim * n_src

        directions = self._generate_isotropic_directions(n_dirs, source_pos, output_pos)
        source_pos = np.array([source_pos.tolist() for _ in range(source_ndim)], dtype=np.float32).reshape(n_dirs,3)
        print('WavePropagator: sources, directions: ', source_pos.shape, directions.shape)
        print('WavePropagator: sources, directions: ', np.unique(source_pos).shape, np.unique(directions.shape))

        # First fast rays propagation without frequency bands
        hits = self.ray_tracer.compute(source_pos, directions)
        ray_inter = hits["geomID"] >= 0
        primID = hits["primID"][ray_inter]
        u = hits["u"][ray_inter]
        v = hits["v"][ray_inter]
        w = 1 - u - v
        mesh_info = acoustic_scene.get_mesh_info()
        print('WavePropagator: mesh_info ', mesh_info.shape)
        inters = (np.vstack(w) * mesh_info[primID][:, 0, :], + np.vstack(u) * mesh_info[primID][:, 1, :], + np.vstack(v) * mesh_info[primID][:, 2, :])
        print('WavePropagator: inters', len(inters))
        
        print('WavePropagator: first fast rays propagation ended', self.combo)

        # Compute Paths for band_idx
        task_tracer = []
        for bands_idx in range(n_bands):
            # Init  RayData storage
            energies = np.full((source_pos.shape[0],1), [1], dtype=np.float32)
            phases = np.full((source_pos.shape[0],1), [0], dtype=np.float32)
            ray_data = RayData(self.source_idx, self.output_idx, bands_idx)
            ray_data.add_data(recursion_idx=self.recursion_idx, n_rays=source_pos.shape[0], origins=source_pos, directions=directions, energies=energies, phases=phases)
            _ = self.entity_manager.register('ray_datas', ray_data)
            task_tracer += [self.diff_path_tracer.compute(hits, bands_idx, ray_data)]
        tracer_results = compute(*task_tracer)

        print('WavePropagator: paths for band computed', self.combo)

        self.recursion_idx += 1
        for next_source_pos, next_directions, bands_idx, ray_data in tracer_results:
            if isinstance(next_source_pos, np.ndarray) and isinstance(next_directions, np.ndarray):
                if not next_source_pos.shape[0] == 0 and not next_directions.shape[0] == 0:
                    self.compute_loop(next_source_pos, next_directions, bands_idx, ray_data)

    def compute_loop(self, source_pos: np.ndarray, directions: np.ndarray, bands_idx: int, ray_data: RayData):
        print('WavePropagator: compute_loop ray tracer begin', self.combo)
        hits = self.ray_tracer.compute(source_pos, directions)
        results = self.diff_path_tracer.compute(hits, bands_idx, ray_data)
        hits_results = results.compute()
        next_source_pos, next_directions, bands_idx, ray_data = hits_results

        print(f"WavePropagator: {self.recursion_idx} compute_loop started", self.combo)
        if isinstance(next_source_pos, np.ndarray) and isinstance(next_directions, np.ndarray):
            if not next_source_pos.shape[0] == 0 and not next_directions.shape[0] == 0:
                self.compute_loop(next_source_pos, next_directions, bands_idx)

        print(f"WavePropagator: compute_loop end", self.combo)

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

    def _generate_isotropic_directions(self, n_dirs: int, source_pos: np.ndarray, dest_pos: np.ndarray, seed: int = 1) -> np.ndarray:
        """
        Evenly distribute rays on 4π steradian from source(s) to destination.
    
        Parameters:
        -----------
        source_pos : numpy.ndarray
            Source position(s) as shape (3,) for single source or (n_sources, 3) for multiple sources
        dest_pos : numpy.ndarray
            Destination position as shape (3,)
        n_dirs : int
            Number of final directions
        
        Returns:
        --------
        numpy.ndarray
            Direction vectors of shape (n_dirs, 3)
        """
        n_sources = source_pos.shape[0]
    
        # Calculate main direction from each source to destination
        main_dirs = dest_pos - source_pos  # shape (n_sources, 3)
    
        # Normalize main directions
        main_dirs_norm = np.linalg.norm(main_dirs, axis=1, keepdims=True)
        main_dirs_norm[main_dirs_norm <= 1e-10] = 1e-10
        main_dirs = main_dirs / main_dirs_norm
    
        # Generate evenly distributed points on sphere using Fibonacci sphere algorithm
        directions = np.zeros((n_dirs, 3), dtype=np.float32)

        # Insert main direction
        directions[:n_sources] = main_dirs
    
        # Golden ratio
        phi = np.pi * (3. - np.sqrt(5.))

        for i in range(n_sources, n_dirs):
            y = 1 - (i / float(n_dirs - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)
        
            theta = phi * i
        
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
        
            directions[i] = [x, y, z]
    
            return directions
