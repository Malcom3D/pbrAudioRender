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
        src_pos, self.source_mesh = self._get_source_mesh(source_idx)
        out_pos, self.output_mesh = self._get_output_mesh(output_idx)

        # get the AcousticDomain mesh
        self.ac_mesh = _acoustic_domain_mesh(config)

        config_objs = config.objects

        return self._build_scene(src_pos, out_pos, config_objs)

    def _build_scene(self, src_pos: np.ndarray, out_pos: np.ndarray, config_objs: Any):
        """
        Build SIMD-friendly scene representation for Embree ray tracing.
        
        Args:
            src_pos: Source position (3D)
            out_pos: Output position (3D)
            config_objs: List of object configurations
            
        Returns:
            Dictionary containing scene data for efficient ray tracing
        """
        embreeDevice = rtc.EmbreeDevice()
        scene = rtcs.EmbreeScene(embreeDevice)

        # Store mesh information for SIMD processing
        self.mesh_info = {
            'vertices': [],      # List of vertex arrays
            'faces': [],         # List of face arrays
            'normals': [],       # List of normal arrays
            'obj_ids': [],       # Object IDs for each mesh
            'materials': [],     # Material properties for each mesh
            'bboxes': [],        # Bounding boxes for each mesh
            'triangle_counts': [], # Triangle counts for each mesh
            'vertex_offsets': [0], # Cumulative vertex offsets
            'face_offsets': [0],   # Cumulative face offsets
        }
        
        # Add acoustic domain mesh (obj_id = -1)
        if self.ac_mesh is not None:
            task_scene = [self._add_mesh_to_scene(scene, self.ac_mesh, -1, "acoustic_domain")]
        
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
        print(meshes_results)

        # Add all acoustic objects with their actual obj_ids
        for obj_mesh, obj_idx, name in meshes_results:
            task_scene += [self._add_mesh_to_scene(scene, obj_mesh, obj_idx, name)]

        # Finalize scene building
        results = compute(*task_scene) 
        
        # Build SIMD-friendly arrays for batch processing
        self._build_simd_arrays()
        
        return scene, self.mesh_info
#        return {
#            'scene': scene,
#            'mesh_info': self.mesh_info,
#            'source_idx': self.combo[0],
#            'output_idx': self.combo[1],
#            'source_pos': src_pos,
#            'output_pos': out_pos
#        }

    @delayed
    def _add_mesh_to_scene(self, scene: rtcs.EmbreeScene, mesh: trimesh.Trimesh, obj_id: int, name: str):
        """
        Add a mesh to the Embree scene and store SIMD-friendly data.
        
        Args:
            scene: Embree scene
            mesh: Trimesh object
            obj_id: Object identifier
            name: Mesh name for debugging
        """
        # Extract mesh data
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32)
        
        # Compute face normals if not present
        if mesh.face_normals is None:
            mesh.fix_normals()
        
        # Add mesh to Embree scene
        embree_mesh = TriangleMesh(scene, vertices, faces)
        
        # Store mesh information for SIMD processing
        self.mesh_info['vertices'].append(vertices)
        self.mesh_info['faces'].append(faces)
        self.mesh_info['normals'].append(mesh.face_normals.astype(np.float32))
        self.mesh_info['obj_ids'].append(np.full(faces.shape[0], obj_id, dtype=np.int32))
        
#        # Get material properties from config
#        material = self._get_material_properties(obj_id)
#        self.mesh_info['materials'].append(material)
        
#        # Store bounding box
#        bbox = mesh.bounds
#        self.mesh_info['bboxes'].append(bbox)
        
        # Store triangle count
        triangle_count = faces.shape[0]
        self.mesh_info['triangle_counts'].append(triangle_count)
        
        # Update offsets
        last_vertex_offset = self.mesh_info['vertex_offsets'][-1]
        last_face_offset = self.mesh_info['face_offsets'][-1]
        self.mesh_info['vertex_offsets'].append(last_vertex_offset + vertices.shape[0])
        self.mesh_info['face_offsets'].append(last_face_offset + triangle_count)
        
#        # Store reference to embree mesh
#        if not hasattr(self, 'embree_meshes'):
#            self.embree_meshes = []
#        self.embree_meshes.append(embree_mesh)

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
                 print('_get_source_mesh: ok' , source_config.name, 'combo: ', self.combo, source_config.type, src_radius)
                 return source_pos, trimesh.creation.icosphere(subdivisions=2, radius=src_radius, transform=[[1, 0, 0, source_pos[0]],[0, 1, 0, source_pos[1]],[0, 0, 1, source_pos[2]],[0, 0, 0, 1]])
         print('_get_source_mesh: no' , source_config.name, 'combo: ', self.combo, source_config.type)
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
