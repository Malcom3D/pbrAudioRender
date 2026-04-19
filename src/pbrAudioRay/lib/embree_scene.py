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
from dask import delayed, compute

import trimesh
from embreex import rtcore as rtc
from embreex import rtcore_scene as rtcs
from embreex.mesh_construction import TriangleMesh

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.functions import _acoustic_domain_mesh, _load_pose

@dataclass
class EmbreeScene:
    """Wrapper for Embree scene with SIMD-friendly data layout"""
    entity_manager: EntityManager
    combo: Tuple[int, int]
    frame_idx: int

    def __post_init__(self):
        config = self.entity_manager.get('config')

        source_idx, output_idx = self.combo
        self.src_pos, self.source_mesh = self._get_source_mesh(source_idx)
        self.out_pos, self.output_mesh = self._get_output_mesh(output_idx)

        # Init store for meshes faces and obj_idx information for SIMD processing
        self.mesh_info = np.zeros((0,3,3), dtype=np.float32)
        self.scene_info = np.array([], dtype=np.int32)
        self.mat_info = {}

        # Get frequency bands
        self.freq_bands = self.entity_manager.get('frequency_bands').get_bands()
        n_bands = len(self.freq_bands)

        # Init store for acoustiic material info
        self.sound_speed = np.zeros((0,1), dtype=np.float32)
        self.density = np.zeros((0,1), dtype=np.float32)
        self.absorption = np.zeros((0,2,n_bands), dtype=np.float32)
        self.refraction = np.zeros((0,2,n_bands), dtype=np.float32)
        self.reflection = np.zeros((0,2,n_bands), dtype=np.float32)
        self.scattering = np.zeros((0,2,n_bands), dtype=np.float32)

        # get the AcousticDomain mesh
        self.ac_mesh = _acoustic_domain_mesh(config)

        # Build embrex scene
        self.scene = self._build_scene(self.src_pos, self.out_pos)

    def _build_scene(self, src_pos: np.ndarray, out_pos: np.ndarray):
        """
        Build SIMD-friendly scene representation for Embree ray tracing.
        
        Args:
            src_pos: Source position (3D)
            out_pos: Output position (3D)
            config_objs: List of object configurations
            
        Returns:
            Dictionary containing scene data for efficient ray tracing
        """
        config = self.entity_manager.get('config')
        config_objs = config.objects

        embreeDevice = rtc.EmbreeDevice()
        scene = rtcs.EmbreeScene(embreeDevice)

        # Add acoustic domain mesh (obj_id = -1)
        if self.ac_mesh is not None:
            task_scene = [self._add_mesh_to_scene(scene, self.ac_mesh, -1, "acoustic_domain", config.acoustic_domain)]
        
        # Add source mesh (obj_id = -2)
        if self.source_mesh is not None:
            task_scene += [self._add_mesh_to_scene(scene, self.source_mesh, -2, "source")]
        
        # Add output mesh (obj_id = -3)
        if self.output_mesh is not None:
            task_scene += [self._add_mesh_to_scene(scene, self.output_mesh, -3, "output")]
        
        # Get all acoustic objects mesh
        task_mesh = []
        objects = self.entity_manager.get('objects')
        for obj_config in config_objs:
            for key in objects.keys():
                if objects[key].obj_idx == obj_config.idx:
                    task_mesh += [self._get_obj_mesh(objects[key], obj_config, src_pos, out_pos)]
        meshes_results = compute(*task_mesh)

        # Add all acoustic objects with their actual obj_ids
        for obj_mesh, obj_idx, name in meshes_results:
            task_scene += [self._add_mesh_to_scene(scene, obj_mesh, obj_idx, name, obj_config)]

        # Finalize scene building
        results = compute(*task_scene) 
        
        return scene

    @delayed
    def _add_mesh_to_scene(self, scene: rtcs.EmbreeScene, mesh: trimesh.Trimesh, obj_idx: int, name: str, obj_config: Any = None):
        """
        Add a mesh to the Embree scene and store SIMD-friendly data.
        
        Args:
            scene: Embree scene
            mesh: Trimesh object
            obj_idx: Object identifier
            name: Mesh name for debugging
            obj_config: Object config
        """
        config = self.entity_manager.get('config')

        if mesh == None:
            return

        # Extract mesh data
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32)
        
        # Compute face normals if not present
        if mesh.face_normals is None:
            mesh.fix_normals()
        
        # Add mesh to Embree scene
        embree_mesh = TriangleMesh(scene, vertices[faces])
        
        # Get triangle count
        triangle_count = faces.shape[0]

        # Store mesh information for SIMD processing
        self.scene_info = np.append(self.scene_info, np.full((faces.shape[0],), obj_idx, dtype=np.int32))
        self.mesh_info = np.append(self.mesh_info, mesh.vertices[mesh.faces], axis=0)

        # Number of frequency bands
        n_bands = len(self.freq_bands)

        # Get Material Info
        sound_speed = obj_config.acoustic_shader.sound_speed
        self.sound_speed = np.append(self.sound_speed, np.full((faces.shape[0],), sound_speed, dtype=np.float32))
        density = obj_config.acoustic_shader.density
        self.density = np.append(self.density, np.full((faces.shape[0],), density, dtype=np.float32))
        if obj_idx >= 0:
            # Get Object AcousticShader
            coeffs, phases = obj_config.acoustic_shader.acoustic_properties.absorption.get_bands_avg(self.freq_bands)
            self.absorption = np.append(self.absorption, np.full((faces.shape[0],2,n_bands), [coeffs.tolist(), phases.tolist()], dtype=np.float32))

            coeffs, phases = obj_config.acoustic_shader.acoustic_properties.refraction.get_bands_avg(self.freq_bands)
            self.refraction = np.append(self.refraction, np.full((faces.shape[0],2,n_bands), [coeffs.tolist(), phases.tolist()], dtype=np.float32))

            coeffs, phases = obj_config.acoustic_shader.acoustic_properties.reflection.get_bands_avg(self.freq_bands)
            self.reflection = np.append(self.reflection, np.full((faces.shape[0],2,n_bands), [coeffs.tolist(), phases.tolist()], dtype=np.float32))

            coeffs, phases = obj_config.acoustic_shader.acoustic_properties.scattering.get_bands_avg(self.freq_bands)
            self.scattering = np.append(self.scattering, np.full((faces.shape[0],2,n_bands), [coeffs.tolist(), phases.tolist()], dtype=np.float32))

    @delayed
    def _get_obj_mesh(self, object: Any, obj_config: Any, src_pos: np.ndarray, out_pos: np.ndarray):
        """ Get mesh object geometry with LOD from AcousticObject """
        mesh = object.get_mesh(self.frame_idx, src_pos, out_pos)
        return mesh, obj_config.idx, obj_config.name

    def _get_source_mesh(self, source_idx: int): 
         config = self.entity_manager.get('config')
         # Build the source mesh
         for src in config.sources:
             if src.idx == source_idx:
                 source_config = src
                 break

         # Get positions and rotations over time
         source_positions, source_rotations = _load_pose(source_config)

         if source_config.static:
             source_pos = source_positions
             source_rot = source_rotations
         else:
             source_pos = source_positions[self.frame_idx]
             source_rot = source_rotations[self.frame_idx]

         # Build the source mesh (icosphere) if it's a spherical source
         if source_config.type == 'SPHERE':
             src_radius = source_config.size
             if src_radius > 0:
                 return source_pos, trimesh.creation.icosphere(subdivisions=2, radius=src_radius, transform=[[1, 0, 0, source_pos[0]],[0, 1, 0, source_pos[1]],[0, 0, 1, source_pos[2]],[0, 0, 0, 1]])
         return source_pos, None

    def _get_output_mesh(self, output_idx: int): 
         config = self.entity_manager.get('config')
         # Build the output mesh
         for out in config.outputs:
             if out.idx == output_idx:
                 output_config = out
                 break

         # Get positions and rotations over time
         output_positions, output_rotations = _load_pose(output_config)

         if output_config.static:
             output_pos = output_positions
             output_rot = output_rotations
         else:
             output_pos = output_positions[self.frame_idx]
             output_rot = output_rotations[self.frame_idx]

         # Build the physical output size as mesh (icosphere) as obj_idx = -3
         out_radius = output_config.size
         if out_radius > 0:
             return output_pos, trimesh.creation.icosphere(subdivisions=2, radius=out_radius, transform=[[1, 0, 0, output_pos[0]],[0, 1, 0, output_pos[1]],[0, 0, 1, output_pos[2]],[0, 0, 0, 1]])
         return output_pos, None
