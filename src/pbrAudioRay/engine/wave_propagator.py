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

import dask
import trimesh
import numpy as np
from dask import delayed, compute
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..engine.interface import InterfaceManager
from ..engine.resonance import Resonance
from ..engine.termination import SimulationTermination

from ..lib.functions import _load_pose, _acoustic_domain_mesh

@dataclass
class WavePropagator:
    """Main wave propagator manager coordinating all physical processes."""
    entity_manager: EntityManager
    combo: Tuple[int, int]
    source_idx: int = None
    output_idx: int = None
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.source_idx = self.combo[0]
        self.output_idx = self.combo[1]
        self.interface_manager = InterfaceManager(self.entity_manager)
        self.resonance = Resonance(self.entity_manager)
        self.termination = SimulationTermination(self.entity_manager)
        
        # Get source and output configs
        self.source_config = None
        self.output_config = None
        for src in config.sources:
            if src.idx == self.source_idx:
                self.source_config = src
                break
        for out in config.outputs:
            if out.idx == self.output_idx:
                self.output_config = out
                break
        
        # Store impulse response (time, frequency bands)
        # We'll compute IR per frame and then interpolate to sample rate
        self.ir = None  # Will be filled after compute
    
        self.objects = []  # List of trimesh objects for each acoustic object
        self.object_ids = []  # Corresponding object indices
        
    def compute(self, frame_idx):
        """Compute impulse response for this source-output pair."""
        # Get positions and rotations over time
        source_positions, source_rotations = _load_pose(self.source_config)
        output_positions, output_rotations = _load_pose(self.output_config)

        if self.source_config.static:
            source_pos = source_positions
            source_rot = source_rotations
        else:
            source_pos = source_positions[frame_idx]
            source_rot = source_rotations[frame_idx]
        if self.output_config.static:
            output_pos = output_positions
            output_rot = output_rotations
        else:
            output_pos = output_positions[frame_idx]
            output_rot = output_rotations[frame_idx]

        # Update the scene at frame_idx
        scene_meshes_ids, scene_meshes = self._update_frame(frame_idx, source_pos, output_pos)

        # Pass data to InterfaceManager
        rays_results = self.interface_manager.compute(frame_idx, source_pos, source_rot, output_pos, output_rot, scene_meshes, scene_meshes_ids)

#        rays_results = self._compute_frame(frame_idx, source_pos, source_rot, output_pos, output_rot, scene_meshes, scene_meshes_ids)
#        self.ir = self._compute_ir(rays_results)

    def _update_frame(self, frame_idx: int, source_pos: np.ndarray, output_pos: np.ndarray):
        """Update acoustic objects for a given frame index."""
        config = self.entity_manager.get('config')

        # Add the acoustic domain as obj_idx = -1
        ac = _acoustic_domain_mesh(config)
        meshes = [ac]
        meshes_ids = [-1]

        # Add the physical source size as mesh (icosphere) as obj_idx = -3 if it's a spherical source
        if self.source_config.type == 'SPHERE':
            src_radius = self.source_config.size
            if src_radius > 0:
                meshes += trimesh.creation.icosphere(subdivisions=2, radius=src_radius, transform=[[1, 0, 0, source_pos[0]],[0, 1, 0, source_pos[1]],[0, 0, 1, source_pos[2]],[0, 0, 0, 1]])
                meshes_ids += [-2]

        # Add the physical output size as mesh (icosphere) as obj_idx = -3
        out_radius = self.output_config.size
        if out_radius > 0:
            meshes += trimesh.creation.icosphere(subdivisions=2, radius=out_radius, transform=[[1, 0, 0, output_pos[0]],[0, 1, 0, output_pos[1]],[0, 0, 1, output_pos[2]],[0, 0, 0, 1]])
            meshes_ids += [-3]

        objects = self.entity_manager.get('objects')
        for obj_config in config.objects:
            if not obj_config.fractured or frame_idx < obj_config.fractured:
                for obj_key in objects.keys():
                    if objects[obj_key].obj_idx == obj_config.idx:
                        mesh = objects[obj_key].get_mesh(frame_idx, source_pos, output_pos)
                        meshes.append(mesh) 
                        meshes_ids.append(obj_config.idx)

        return meshes_ids, meshes
        
    def _compute_ir(self, rays):
        """Compute rays into an impulse response."""
        # We have a list of rays. Each ray has a delay (time = length / speed_of_sound)
        # and an amplitude (energy). We'll accumulate contributions in time bins.
        # We'll use a simple histogram approach.
        config = self.entity_manager.get('config')
        speed_of_sound = config.acoustic_domain.acoustic_shader.sound_speed
        # We'll sample IR at sample rate
        # find length of IR
        max_length = 0
        for ray in rays:
            if not ray == None:
                max_length = max(ray.length, max_length)
        samples = 1 + int(max_length * self.sample_rate / speed_of_sound)
        ir_amp = np.zeros(samples)
        
        for ray in rays:
            if not ray == None:
                delay = ray.length / speed_of_sound
                # Find nearest sample
                sample_idx = int(delay * self.sample_rate)
                if 0 <= sample_idx < len(ir_amp):
                    ir_amp[sample_idx] += ray.energy  # assume energy is amplitude
        return ir_amp
    
    def get_impulse_response(self):
        """Return the computed impulse response (time, amplitude) for this source-output pair."""
        if self.ir is None:
            raise RuntimeError("Compute first")
        return self.ir
