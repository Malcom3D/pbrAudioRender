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
from ..lib.acoustic_scene import AcousticScene

@dataclass
class EmbreeScene:
    """Wrapper for Embree scene with SIMD-friendly data layout"""
    entity_manager: EntityManager
    combo: Tuple[int, int]
    frame_idx: int

    def __post_init__(self):
        config = self.entity_manager.get('config')

        # Get frequency bands
        self.freq_bands = self.entity_manager.get('frequency_bands').get_bands()

        # Init store for multi bands fraquencies scene information for SIMD processing
        self.acoustic_scene = AcousticScene(self.freq_bands)

        # get source and output mesh
        source_idx, output_idx = self.combo
        self.src_pos, source_mesh = self._get_source_mesh(source_idx)
        self.out_pos, output_mesh = self._get_output_mesh(output_idx)

        # Build embrex scene
        self.scene = self._build_scene(source_mesh, output_mesh)

    def _build_scene(self, source_mesh: trimesh.Trimesh, output_mesh: trimesh.Trimesh):
        """
        Build SIMD-friendly scene representation for Embree ray tracing.
        
        Args:
            source_mesh: trimesh.Trimesh source mesh
            output_mesh: trimesh.Trimesh output mesh
            
        Returns:
            Dictionary containing scene data for efficient ray tracing
        """
        config = self.entity_manager.get('config')
        config_objs = config.objects
        num_objs = len(config_objs)

        # get the AcousticDomain mesh
        ac_mesh = _acoustic_domain_mesh(config)

        # Add acoustic domain mesh (obj_id = -1)
#        src_medium, out_medium = (np.nan for _ in range(2))
        if ac_mesh is not None:
#            if ac_mesh.contains(self.src_pos.reshape(-1,3)):
#                # Store mesh information for SIMD processing
#                src_medium = -1
#                if isinstance(source_mesh, trimesh.Trimesh) and hasattr(source_mesh.metadata, 'radius'):
#                    src_radius = mesh.metadata['radius']
#                self.acoustic_scene.add_aso_info(-2, self.src_pos, src_medium, src_radius)
#            if ac_mesh.contains(self.out_pos.reshape(1,3)) and out_medium == np.nan:
#                out_medium = -1
#                if isinstance(output_mesh, trimesh.Trimesh) and hasattr(output_mesh.metadata, 'radius'):
#                    out_radius = mesh.metadata['radius']
#                self.acoustic_scene.add_aso_info(-3, self.out_pos, out_medium, out_radius)
            task_scene = [self._add_mesh_to_scene(ac_mesh, -1, "acoustic_domain", config.acoustic_domain)]
            num_objs += 1
        
        # Add source mesh (obj_id = -2)
        if source_mesh is not None:
            task_scene += [self._add_mesh_to_scene(source_mesh, -2, "source")]
            num_objs += 1
        
        # Add output mesh (obj_id = -3)
        if output_mesh is not None:
            task_scene += [self._add_mesh_to_scene(output_mesh, -3, "output")]
            num_objs += 1

#        self.acoustic_scene.set_num_objs(num_objs)

        # Get all acoustic objects mesh
        task_mesh = []
        objects = self.entity_manager.get('objects')
        for obj_config in config_objs:
            for key in objects.keys():
                if objects[key].obj_idx == obj_config.idx:
                    task_mesh += [self._get_obj_mesh(objects[key], obj_config)]
        meshes_results = compute(*task_mesh)

        for obj_mesh, obj_idx, name in meshes_results:
#            # Check if source and output points is inside the mesh and add all acoustic objects with their actual obj_ids
#            if obj_mesh.is_watertight:
#                if obj_mesh.contains(self.src_pos.reshape(1,3)) and src_medium == np.nan:
#                    # Store mesh information for SIMD processing
#                    src_medium = obj_idx
#                    if isinstance(source_mesh, trimesh.Trimesh) and hasattr(source_mesh.metadata, 'radius'):
#                        src_radius = mesh.metadata['radius']
#                    self.acoustic_scene.add_aso_info(-2, self.src_pos, src_medium, src_radius)
#                if obj_mesh.contains(self.out_pos.reshape(1,3)) and out_medium == np.nan:
#                    out_medium = obj_idx
#                    if isinstance(output_mesh, trimesh.Trimesh) and hasattr(output_mesh.metadata, 'radius'):
#                        out_radius = mesh.metadata['radius']
#                    self.acoustic_scene.add_aso_info(-3, self.out_pos, out_medium, out_radius)

            task_scene += [self._add_mesh_to_scene(obj_mesh, obj_idx, name, obj_config)]

        # Finalize scene building
        results = compute(*task_scene) 
        
        # Get mesh_info
        mesh_info = self.acoustic_scene.mesh_info
        
        # Init embree scene
#        embreeDevice = rtc.EmbreeDevice()
#        scene = rtcs.EmbreeScene(embreeDevice)
        scene = rtcs.EmbreeScene()

        # Add meshes to Embree scene
        embree_meshes = TriangleMesh(scene, mesh_info)
        
        return scene

    @delayed
    def _add_mesh_to_scene(self, mesh: trimesh.Trimesh, obj_idx: int, name: str, obj_config: Any = None):
        """
        Add a mesh to the Embree scene and store SIMD-friendly data.
        
        Args:
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
        
        # Store mesh information for SIMD processing
        self.acoustic_scene.add_mesh_info(obj_idx, obj_config, vertices, faces)

    @delayed
    def _get_obj_mesh(self, object: Any, obj_config: Any):
        """ Get mesh object geometry with LOD from AcousticObject """
        mesh = object.get_mesh(self.frame_idx, self.src_pos, self.out_pos)
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
