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

import sys
import math
import copy 
import numpy as np
import numba as nb
import trimesh
from numba import prange
from dask import delayed, compute
from typing import List, Tuple
from dataclasses import dataclass

from pbrAudioRay.core.entity_manager import EntityManager
from pbrAudioRay.engine.ray_tracer import AcousticRayTracer

from pbrAudioRay.lib.functions import _load_pose
from pbrAudioRay.lib.ray_data import RayData
from pbrAudioRay.lib.geometry_data import GeometryData
from pbrAudioRay.lib.material_properties import MaterialProperties
from pbrAudioRay.lib.medium_properties import MediumProperties

@dataclass
class WavePropagator:
    """Wave propagator using SIMD and parallel processing"""
    entity_manager: EntityManager
    combo: Tuple[int, int]
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        max_interactions = config.wave_propagation.max_interactions
        sys.setrecursionlimit(max_interactions*2)

        # Initialize Ray data
        self.ray_data = RayData()

        # Get scene data
        self.geometry_data = self.entity_manager.get('geometry_data')
        self.material_properties = self.entity_manager.get('material_properties')
        self.medium_properties = self.entity_manager.get('medium_properties')

        # Finalize scene
        self._finalize_scene()

    def _finalize_scene(self):
        source_idx, output_idx = self.combo
        self._initialize_sources(source_idx)
        self._initialize_outputs(output_idx)

    def _initialize_sources(self, source_idx: int):
        """Initialize source positions and directions."""
        config = self.entity_manager.get('config')
        n_rays = config.system.number_of_rays

        for src_config in config.sources:
            if src_config.idx == source_idx:
                pose = np.load(f"{src_config.pose_path}/{src_config.name}.npz")
                source_pos = pose[pose.files[0]].reshape(-1, 3)

                source_arr = np.full((n_rays, 3), [source_pos], dtype=np.float32)
                self.ray_data.origins = np.append(self.ray_data.origins, source_arr, axis=0)
        
    def _initialize_outputs(self, output_idx: int):
        """Initialize output positions."""
        config = self.entity_manager.get('config')
        frequency_bands = self.entity_manager.get('frequency_bands')
        n_bands = len(frequency_bands.get_bands())
        n_rays = config.system.number_of_rays

        for out_config in config.outputs:
            if out_config.idx == output_idx:
                pose = np.load(f"{out_config.pose_path}/{out_config.name}.npz")
                self.ray_data.destinations = pose[pose.files[0]].reshape(-1, 3)

            # Create output sphere geometry
            if out_config.size == 0:
                out_config.size = 0.1

            mesh = trimesh.creation.icosphere(subdivisions=2, radius=out_config.size)
            mesh.apply_transform([
                [1, 0, 0, self.ray_data.destinations[0][0]],
                [0, 1, 0, self.ray_data.destinations[0][1]],
                [0, 0, 1, self.ray_data.destinations[0][2]],
                [0, 0, 0, 1]
            ])

            vertices = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.int32)

            self.geometry_data.mesh_info = np.append(self.geometry_data.mesh_info, mesh.vertices[mesh.faces], axis=0)
            self.geometry_data.scene_info = np.append(self.geometry_data.scene_info, np.full((mesh.vertices[mesh.faces].shape[0],), [-3], dtype=np.int32))

            # Add null properties for output geometry
            n_faces = vertices[faces].shape[0]
            self.material_properties.roughness = np.append(self.material_properties.roughness, np.full((n_faces, 1), 0.0, dtype=np.float32), axis=0)

            for prop_name in ['absorption', 'reflection', 'refraction', 'scattering']:
                coeffs = getattr(self.material_properties, f'{prop_name}_coeffs')
                phases = getattr(self.material_properties, f'{prop_name}_phases')

                new_coeffs = np.full((n_faces, n_bands),  1.0 if prop_name == 'absorption' else 0.0, dtype=np.float32)
                new_phases = np.full((n_faces, n_bands), 0.0, dtype=np.float32)

                setattr(self.material_properties, f'{prop_name}_coeffs', np.append(coeffs, new_coeffs, axis=0))
                setattr(self.material_properties, f'{prop_name}_phases', np.append(phases, new_phases, axis=0))
    @delayed
    def compute(self, frame_idx):
        """Compute impulse response for a single frame"""
        frequency_bands = self.entity_manager.get('frequency_bands')
        n_bands = len(frequency_bands.get_bands())

        tracer_task = []
        for bands_idx in range(n_bands):
            ray_data = copy.deepcopy(self.ray_data)
            ray_data.bands_idx = bands_idx
            ray_tracer = AcousticRayTracer(self.entity_manager, self.geometry_data, self.material_properties, self.medium_properties, ray_data)
            tracer_task += [ray_tracer.compute()]

        results = compute(*tracer_task)
        #return output_data
        for output_data in results:
            print(f"Wave propagator {self.combo} bands_idx {output_data.bands_idx} ended")
