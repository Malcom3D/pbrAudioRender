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

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import dask
from dask import delayed, compute

from ..core.entity_manager import EntityManager
from ..engine.interface import InterfaceManager
from ..engine.resonance import Resonance
from ..engine.termination import SimulationTermination

from ..engine.ray_tracer import RayTracer
from ..lib.ray_data import RayData
from ..lib.functions import _load_pose

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
        self.ray_tracer = RayTracer(self.entity_manager)
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
        
        # Frequency bands for impulse response
        frequency_bands = self.entity_manager.get('frequency_bands')

        # Get sample rate
        self.sample_rate = config.system.sample_rate
        
        # Store impulse response (time, frequency bands)
        # We'll compute IR per frame and then interpolate to sample rate
        self.ir = None  # Will be filled after compute
    
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

        # Update ray_tracer scene for frame_idx
        self.ray_tracer.update_frame(frame_idx, source_pos, output_pos)

        all_rays = self._compute_frame(frame_idx, source_pos, source_rot, output_pos, output_rot)
        self.ir = self._compute_ir(all_rays)
        
    def _compute_frame(self, frame_idx, source_pos, source_rot, output_pos, output_rot):
        """Compute ray paths for a single frame."""
        # Direct path: source to output
        direct_ray = self._trace_direct_path(source_pos, output_pos)
        # Reverse path: output to source (for diffraction)
        reverse_ray = self._trace_reverse_path(output_pos, source_pos)
        # Also we need to consider reflections/refractions from objects
        reflected_rays = self._trace_reflected_paths(source_pos, output_pos)
        
        # Combine all rays
        all_rays = [direct_ray] + reverse_ray + reflected_rays
        # Apply frequency-dependent attenuation to each ray
        for ray in all_rays:
            if ray:
                self._apply_attenuation(ray, source_pos, output_pos)
        return all_rays
    
    def _trace_direct_path(self, src, dst):
        """Trace direct line-of-sight path."""
        direction = dst - src
        dist = np.linalg.norm(direction)
        if dist == 0:
            return None
        direction = direction / dist
        # Check if path is blocked by any object
        hit = self.ray_tracer.intersect_ray(src, direction, max_distance=dist)
        if hit['hit']:
            # If blocked, no direct path
            return None
        # Create ray data
        ray = RayData(
            origin=src,
            direction=direction,
            length=dist,
            energy=1.0,  # initial energy
            reflection_count=0,
            path=[src, dst]
        )
        return ray
    
    def _trace_reverse_path(self, src, dst):
        """Trace reverse path (listener to source) for diffraction."""
        # Similar to direct but from listener to source
        # For diffraction, we might need to generate multiple rays around edges
        # For simplicity, we'll just return direct reverse path for now
        direction = dst - src
        dist = np.linalg.norm(direction)
        if dist == 0:
            return []
        direction = direction / dist
        hit = self.ray_tracer.intersect_ray(src, direction, max_distance=dist)
        if hit['hit']:
            # Path blocked, but we might still have diffracted paths
            # For now, return empty
            return []
        ray = RayData(
            origin=src,
            direction=direction,
            length=dist,
            energy=1.0,
            reflection_count=0,
            path=[src, dst]
        )
        return [ray]
    
    def _trace_reflected_paths(self, src, dst):
        """Trace reflected/refracted paths using recursive ray tracing."""
        # We'll use ray tracing with recursion depth limited by max_reflection
        config = self.entity_manager.get('config')
        max_reflection = config.interface.max_reflection
        # We'll generate rays from source, bounce, and check if they hit the listener
        rays = []
        self._trace_recursive(src, dst, direction=None, depth=0, max_depth=max_reflection, ray_so_far=None, rays_list=rays)
        return rays
    
    def _trace_recursive(self, current_pos, target, direction, depth, max_depth, ray_so_far, rays_list):
        """Recursive ray tracing to find paths from current_pos to target."""
        if depth >= max_depth:
            return
        # If we have a direction, we continue; else we start by sampling directions
        # For simplicity, we'll use a simple approach: at each intersection, generate reflected and refracted rays.
        # We need to know the object properties at intersection.
        # For now, we'll skip detailed implementation and return empty.
        # This will be expanded later.
        pass
    
    def _apply_attenuation(self, ray, src, dst):
        """Apply frequency-dependent attenuation due to propagation and interactions."""
        # We'll compute attenuation per frequency band
        # For each interaction along the path, apply absorption/reflection coefficients
        # For now, just a placeholder
        pass

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
        samples = int(max_length * self.sample_rate / speed_of_sound)
        ir_amp = np.zeros(samples)
        
        for ray in rays:
            if ray is None:
                continue
            delay = ray.length / speed_of_sound
            # Find nearest sample
            sample_idx = int(delay * self.sample_rate)
            if 0 <= sample_idx < len(ir_amp):
                print('_compute_ir: ', self.combo, sample_idx, ray.energy)
                ir_amp[sample_idx] += ray.energy  # assume energy is amplitude
        return ir_amp
    
    def get_impulse_response(self):
        """Return the computed impulse response (time, amplitude) for this source-output pair."""
        if self.ir is None:
            raise RuntimeError("Compute first")
        return self.ir
