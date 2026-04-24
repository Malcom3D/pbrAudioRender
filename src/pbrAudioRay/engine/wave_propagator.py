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
        n_points = 0
        for source in self.config.sources:
            if source.idx == self.source_idx:
                if source.type == 'SPERE' and source.size > 0:
                    source_size = source.size
                    n_points = int(np.random.uniform(1, 10, size=1))
                    source_pos = self._source_points(n_points, source_pos, source_size)

        n_src = source_pos.shape[0]
        source_ndim = int(n_rays / n_src)
        n_dirs = source_ndim * n_src

        if source_pos.ndim == 1:
            source_pos = np.array([source_pos.reshape(1,3).tolist() for _ in range(n_dirs)], dtype=np.float32).reshape(n_dirs,3)
        else:
            source_pos = np.array([source_pos.tolist() for _ in range(source_ndim)], dtype=np.float32).reshape(n_dirs,3)

        directions = self._generate_isotropic_directions(n_dirs, source_pos, output_pos)
#        directions = directions[:source_pos.shape[0]]

        # First fast rays propagation without frequency bands
        print('self.ray_tracer.compute: ', n_src, source_pos.shape, directions.shape)
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
        # Ensure source_pos is at least 2D
        if source_pos.ndim == 1:
            source_pos = source_pos.reshape(1, -1)
    
        n_sources = source_pos.shape[0]
    
        # Calculate main direction from each source to destination
        main_dirs = dest_pos - source_pos  # shape (n_sources, 3)
    
        # Normalize main directions
        main_dirs_norm = np.linalg.norm(main_dirs, axis=1, keepdims=True)
        main_dirs = main_dirs / main_dirs_norm
    
        # Generate evenly distributed points on sphere using Fibonacci sphere algorithm
        directions = np.zeros((n_dirs, 3))
    
        # Golden ratio
        phi = np.pi * (3. - np.sqrt(5.))
    
        for i in range(n_dirs):
            y = 1 - (i / float(n_dirs - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)
        
            theta = phi * i
        
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
        
            directions[i] = [x, y, z]
    
        # If we have multiple sources, we need to distribute rays among them
        if n_sources > 1:
            # Distribute rays evenly among sources
            rays_per_source = n_dirs // n_sources
            remainder = n_dirs % n_sources
        
            result_directions = np.zeros((n_dirs, 3))
            start_idx = 0
        
            for source_idx in range(n_sources):
                # Calculate how many rays for this source
                if source_idx < remainder:
                    n_rays_this_source_source = rays_per_source + 1
                else:
                    n_rays_this_source = rays_per_source
            
                if n_rays_this_source > 0:
                    # Generate directions for this source
                    source_dirs = np.zeros((n_rays_this_source, 3))
                
                    # Use Fibonacci sphere for this subset
                    phi_source = np.pi * (3. - np.sqrt(5.))
                    for i in range(n_rays_this_source):
                        y = 1 - (i / float(n_rays_this_source - 1)) * 2
                        radius = np.sqrt(1 - y * y)
                        theta = phi_source * i
                        x = np.cos(theta) * radius
                        z = np.sin(theta) * radius
                        source_dirs[i] = [x, y, z]

                    # Create rotation matrix to align with main direction
                    main_dir = main_dirs[source_idx]

                    # Find rotation axis and angle
                    up = np.array([0, 1, 0])
                    if np.abs(np.dot(main_dir, up)) > 0.99:
                        up = np.array([1, 0, 0])
                
                    # Calculate rotation matrix
                    v = np.cross(up, main_dir)
                    c = np.dot(up, main_dir)
                    s = np.linalg.norm(v)
                
                    if s > 1e-10:
                        v = v / s
                        vx, vy, vz = v
                    
                        # Rodrigues' rotation formula
                        R = np.array([
                            [c + vx*vx*(1-c), vx*vy*(1-c) - vz*s, vx*vz*(1-c) + vy*s],
                            [vy*vx*(1-c) + vz*s, c + vy*vy*(1-c), vy*vz*(1-c) - vx*s],
                            [vz*vx*(1-c) - vy*s, vz*vy*(1-c) + vx*s, c + vz*vz*(1-c)]
                        ])
                    
                        # Apply rotation
                        source_dirs = source_dirs @ R.T
                
                    result_directions[start_idx:start_idx + n_rays_this_source] = source_dirs
                    start_idx += n_rays_this_source
            return result_directions
        else:
            # Single source - align all directions with the main direction
            main_dir = main_dirs[0]
        
            # Create rotation matrix to align [0, 1, 0] with main_dir
            up = np.array([0, 1, 0])
            if np.abs(np.dot(main_dir, up)) > 0.99:
                up = np.array([1, 0, 0])
        
            v = np.cross(up, main_dir)
            c = np.dot(up, main_dir)
            s = np.linalg.norm(v)
        
            if s > 1e-10:
                v = v / s
                vx, vy, vz = v
            
                R = np.array([
                    [c + vx*vx*(1-c), vx*vy*(1-c) - vz*s, vx*vz*(1-c) + vy*s],
                    [vy*vx*(1-c) + vz*s, c + vy*vy*(1-c), vy*vz*(1-c) - vx*s],
                    [vz*vx*(1-c) - vy*s, vz*vy*(1-c) + vx*s, c + vz*vz*(1-c)]
                ])
            
                directions = directions @ R.T
            return directions
