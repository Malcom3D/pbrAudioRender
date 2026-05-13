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
import sys
import math
import copy 
import numpy as np
import numba as nb
import trimesh
import soundfile as sf
from numba import prange
from dask import delayed, compute
from typing import List, Tuple, Any
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

            for prop_name in ['absorption', 'reflection', 'transmission', 'scattering']:
                coeffs = getattr(self.material_properties, f'{prop_name}_coeffs')
                phases = getattr(self.material_properties, f'{prop_name}_phases')

                new_coeffs = np.full((n_faces, n_bands),  1.0 if prop_name == 'absorption' else 0.0, dtype=np.float32)
                new_phases = np.full((n_faces, n_bands), 0.0, dtype=np.float32)

                setattr(self.material_properties, f'{prop_name}_coeffs', np.append(coeffs, new_coeffs, axis=0))
                setattr(self.material_properties, f'{prop_name}_phases', np.append(phases, new_phases, axis=0))
    @delayed
    def compute(self, frame_idx: int):
        """Compute impulse response for a single frame"""
        frequency_bands = self.entity_manager.get('frequency_bands')
        n_bands = len(frequency_bands.get_bands())

        tracer_task = []
        for bands_idx in range(n_bands):
            ray_data = copy.deepcopy(self.ray_data)
            ray_data.bands_idx = bands_idx
            ray_tracer = AcousticRayTracer(self.entity_manager, self.geometry_data, self.material_properties, self.medium_properties, ray_data)
            tracer_task += [ray_tracer.compute(frame_idx)]

        results = compute(*tracer_task)

        # compute x bands_idx IRs for wave_propagators[index].combo
        for output_data in results:
            self._compute_and_save_ir(output_data, frame_idx)
            print(f"Wave propagator {self.combo} bands_idx {output_data.bands_idx} ended")

    def _compute_and_save_ir(self, output_data: Any, frame_idx: int):
        config = self.entity_manager.get('config')
        frequency_bands = self.entity_manager.get('frequency_bands')
        sample_rate = int(config.system.sample_rate)
        source_idx, output_idx = self.combo
        bands_idx = output_data.bands_idx

        # Sort output by delay
        sort_idx = np.argsort(output_data.delay.flatten())

        delay = output_data.delay.flatten()[sort_idx]
        energies = output_data.energies.flatten()[sort_idx]
        phases = output_data.phases.flatten()[sort_idx]
        directions = output_data.directions[sort_idx]

        # Convert delay to samples
        delay_samples = np.round(delay * sample_rate).astype(int)

        # Determine max IR length
        ir_length = int(np.ceil(np.max(delay_samples))) + 10

        # Get ambisonic order for this output
        ambisonic_order = 1  # default
        for out_config in config.outputs:
            if out_config.idx == output_idx:
                ambisonic_order = out_config.order

                n_channels = (ambisonic_order + 1) ** 2
                ambisonics_ir = np.zeros((n_channels, ir_length), dtype=np.float32)

                # Compute complex amplitudes
                complex_amplitudes = np.sqrt(energies) * np.exp(1j * phases)

                # Convert directions to spherical coordinates
                x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
                theta = np.arctan2(y, x)  # Azimuth
                phi = np.arcsin(z)  # Elevation

                # Compute spherical harmonics for each order
                self._compute_spherical_harmonics(ambisonics_ir, delay_samples, complex_amplitudes, theta, phi, ambisonic_order)

                # Apply windowing
                window = np.hanning(ir_length)
                for ch in range(n_channels):
                    ambisonics_ir[ch] *= window

                # Normalize
                max_val = np.max(np.abs(ambisonics_ir))
                if max_val > 0:
                    ambisonics_ir /= max_val

                # Save impulse response
                output_dir = f"{config.system.cache_path}/impulse_responses"
                os.makedirs(output_dir, exist_ok=True)

                filename = f"{output_dir}/ambiIR_{ambisonic_order}_{frame_idx:05}_{source_idx}_{output_idx}_{bands_idx:05}.wav"
                sf.write(filename, ambisonics_ir.T, sample_rate, subtype='FLOAT')

                print(f"Saved ambisonic IR: {filename} for source {source_idx}, output {output_idx}, bands {frequency_bands.get_bands()[bands_idx]} frame {frame_idx}.")
                print(f"  Shape: {ambisonics_ir.T.shape} (samples, channels)")
                print(f"  Duration: {ir_length / sample_rate:.2f} seconds")

    def _compute_spherical_harmonics(self, ambisonics_ir: np.ndarray, delay_samples: np.ndarray, complex_amplitudes: np.ndarray, theta: np.ndarray, phi: np.ndarray, ambisonic_order: int):
        """Compute spherical harmonics and add to IR buffer."""
        n_rays = len(delay_samples)

        if ambisonic_order >= 0:
            # Order 0: W channel (omnidirectional)
            Y_00 = 1.0 / np.sqrt(4 * np.pi)
            for i in range(n_rays):
                sample_idx = delay_samples[i]
                ambisonics_ir[0, sample_idx] += np.real(complex_amplitudes[i] * Y_00)

        if ambisonic_order >= 1:
            # Order 1: X, Y, Z channels (ACN ordering)
            Y_1n1 = np.sqrt(3/(4*np.pi)) * np.sin(theta) * np.cos(phi)  # Y
            Y_10 = np.sqrt(3/(4*np.pi)) * np.sin(phi)                   # Z
            Y_11 = np.sqrt(3/(4*np.pi)) * np.cos(theta) * np.cos(phi)   # X

            for i in range(n_rays):
                sample_idx = delay_samples[i]
                ambisonics_ir[1, sample_idx] += np.real(complex_amplitudes[i] * Y_1n1[i])
                ambisonics_ir[2, sample_idx] += np.real(complex_amplitudes[i] * Y_10[i])
                ambisonics_ir[3, sample_idx] += np.real(complex_amplitudes[i] * Y_11[i])

        if ambisonic_order >= 2:
            # Order 2: Additional 5 channels
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            sqrt_15_4pi = np.sqrt(15/(4*np.pi))

            Y_2n2 = sqrt_15_4pi * sin_theta * cos_theta * cos_phi**2      # R
            Y_2n1 = sqrt_15_4pi * sin_theta * sin_phi * cos_phi           # S
            Y_20 = np.sqrt(5/(16*np.pi)) * (3*sin_phi**2 - 1)              # T
            Y_21 = sqrt_15_4pi * cos_theta * sin_phi * cos_phi             # U
            Y_22 = sqrt_15_4pi * (cos_theta**2 - sin_theta**2) * cos_phi**2  # V

            for i in range(n_rays):
                sample_idx = delay_samples[i]
                ambisonics_ir[4, sample_idx] += np.real(complex_amplitudes[i] * Y_2n2[i])
                ambisonics_ir[5, sample_idx] += np.real(complex_amplitudes[i] * Y_2n1[i])
                ambisonics_ir[6, sample_idx] += np.real(complex_amplitudes[i] * Y_20[i])
                ambisonics_ir[7, sample_idx] += np.real(complex_amplitudes[i] * Y_21[i])
                ambisonics_ir[8, sample_idx] += np.real(complex_amplitudes[i] * Y_22[i])
