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

import os
import numpy as np
import trimesh
from typing import Tuple, Optional, List, Any
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.functions import _load_mesh

@dataclass
class AcousticObject:
    """ Handle mesh object geometry and modal model analysis """
    entity_manager: EntityManager
    config_obj: Any
    obj_idx: int = None

    def __post_init__(self):
        self.obj_idx = self.config_obj.idx

    def get_mesh(self, frame_idx: int, source_pos: np.ndarray = None, output_pos: np.ndarray = None) -> trimesh.Trimesh:
        """ Return mesh object geometry refined using ADR if distance from sources and listeners is greater than a threshold """
        vertices, normals, faces = _load_mesh(self.config_obj, frame_idx)
        mesh = trimesh.Trimesh(vertices=vertices, vertex_normals=normals, faces=faces)

        config = self.entity_manager.get('config')
        adr_threshold = config.system.adr_threshold
        use_extended_reaction = config.wave_propagation.use_extended_reaction

        # If no ADR threshold or no source_pos or output_pos, return full mesh
        if adr_threshold == None or source_pos == None or output_pos == None:
            return mesh

        # Calculate distances from object to source and output
        # Use object's bounding sphere center as reference point
        obj_center = mesh.bounding_sphere.center

        # Calculate minimum distance to any source or output
        source_dist = np.linalg.norm(obj_center - source_pos)
        output_dist = np.linalg.norm(obj_center - output_pos)
        min_distance = min(source_dist, output_dist)

        if distance <= adr_threshold:
            return mesh
        
        # Determine ADR refinement level based on distance
        refinement = self._calculate_adr_level(min_distance, adr_threshold)
        simplified = self._adaptive_refine(mesh, max_refinement=refinement)

        # Return appropriate LOD mesh
        if simplified.is_watertight and simplified.is_volume and simplified.is_winding_consistent:
            if use_extended_reaction:
                # save resonance_obj for pym2f.compute(obj_idx)
                pass
            return simplified
        else:
            return mesh

    def _calculate_adr_level(self, distance: float, adr_threshold: float) -> int:
        """
        Calculate ADR level based on distance.
        
        Args:
            distance: Minimum distance to source/output
            adr_threshold: Base threshold for LOD switching
        
        Returns:
            ADR level (0 = highest detail, >1 lowest detail)
        """
        # Calculate how many threshold multiples we're beyond
        threshold_multiples = distance / adr_threshold
        
        return int(threshold_multiples - 1)

    def _adaptive_refine(mesh, curvature_threshold=0.1, max_refinement=2):
        """
        Refine mesh based on curvature
        """
        mesh_copy = mesh.copy()
    
        for _ in range(max_refinement):
            # Compute discrete curvature at vertices
            curvature = self._compute_curvature(mesh_copy)
        
            # Find edges to split
            edges_to_split = []
            for face in mesh_copy.faces:
                for i in range(3):
                    v1, v2 = face[i], face[(i+1)%3]
                    edge_key = tuple(sorted((v1, v2)))
                
                    # Check if edge needs refinement
                    if (curvature[v1] > curvature_threshold or 
                        curvature[v2] > curvature_threshold):
                        edges_to_split.append(edge_key)
        
            if not edges_to_split:
                break
        
            # Split edges
            mesh_copy = self._split_edges(mesh_copy, list(set(edges_to_split)))
    
        return mesh_copy

    def _compute_curvature(mesh):
        """Compute approximate curvature at vertices"""
        curvature = np.zeros(len(mesh.vertices))
    
        # Build vertex adjacency
        vertex_faces = defaultdict(list)
        for i, face in enumerate(mesh.faces):
            for v in face:
                vertex_faces[v].append(i)
    
        # Compute angle defect as curvature approximation
        for v_idx in range(len(mesh.vertices)):
            face_indices = vertex_faces[v_idx]
            if len(face_indices) < 3:
                continue
        
            total_angle = 0
            for f_idx in face_indices:
                face = mesh.faces[f_idx]
                # Find position of v_idx in face
                pos = np.where(face == v_idx)[0][0]
                v1 = mesh.vertices[face[(pos-1)%3]]
                v2 = mesh.vertices[face[(pos+1)%3]]
            
                # Compute angle at vertex
                vec1 = v1 - mesh.vertices[v_idx]
                vec2 = v2 - mesh.vertices[v_idx]
                angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10))
                total_angle += angle
        
            # Angle defect (2π - sum of angles)
            curvature[v_idx] = max(0, 2 * np.pi - total_angle)
    
        return curvature

    def _split_edges(mesh, edges):
        """Split specified edges by adding vertices at midpoints"""
        vertices = mesh.vertices.tolist()
        faces = mesh.faces.tolist()
    
        # Map for new vertices on edges
        edge_vertices = {}
    
        # First pass: create new vertices
        for v1, v2 in edges:
            if (v1, v2) in edge_vertices or (v2, v1) in edge_vertices:
                continue
        
            # Create new vertex at midpoint
            new_vertex = (np.array(vertices[v1]) + np.array(vertices[v2])) / 2
            new_idx = len(vertices)
            vertices.append(new_vertex)
            edge_vertices[(v1, v2)] = new_idx
            edge_vertices[(v2, v1)] = new_idx
    
        # Second pass: split faces
        new_faces = []
        for face in faces:
            # Check which edges of this face are split
            split_edges_in_face = []
            for i in range(3):
                v1, v2 = face[i], face[(i+1)%3]
                if (v1, v2) in edge_vertices:
                    split_edges_in_face.append(i)
        
            if len(split_edges_in_face) == 0:
                # No split, keep original face
                new_faces.append(face)
            elif len(split_edges_in_face) == 1:
                # One edge split - create two triangles
                i = split_edges_in_face[0]
                v1, v2, v3 = face[i], face[(i+1)%3], face[(i+2)%3]
                new_v = edge_vertices[(v1, v2)]
            
                new_faces.append([v1, new_v, v3])
                new_faces.append([new_v, v2, v3])
            elif len(split_edges_in_face) == 2:
                # Two edges split - create three triangles
                # Find the vertex opposite to the unsplit edge
                unsplit_edge = (set([0,1,2]) - set(split_edges_in_face)).pop()
                v_opposite = face[unsplit_edge]
            
                # Get the two split vertices
                v1 = face[split_edges_in_face[0]]
                v2 = face[(split_edges_in_face[0]+1)%3]
                v3 = face[split_edges_in_face[1]]
                v4 = face[(split_edges_in_face[1]+1)%3]
            
                new_v1 = edge_vertices[(v1, v2)]
                new_v2 = edge_vertices[(v3, v4)]
            
                new_faces.append([v_opposite, v1, new_v1])
                new_faces.append([v_opposite, new_v1, new_v2])
                new_faces.append([v_opposite, new_v2, v3])
            else:
                # All edges split - create four triangles
                v1, v2, v3 = face
                new_v1 = edge_vertices[(v1, v2)]
                new_v2 = edge_vertices[(v2, v3)]
                new_v3 = edge_vertices[(v3, v1)]
            
                new_faces.append([v1, new_v1, new_v3])
                new_faces.append([v2, new_v2, new_v1])
                new_faces.append([v3, new_v3, new_v2])
                new_faces.append([new_v1, new_v2, new_v3])
    
        return trimesh.Trimesh(vertices=vertices, faces=new_faces)
