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
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class RayData:
    src_idx: int
    out_idx: int
    bands_idx: int
    interactions: int = None
    recursions: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.int32))
    origins: np.ndarray = field(default_factory=lambda: np.zeros((0,3), dtype=np.float32))
    directions: np.ndarray = field(default_factory=lambda: np.zeros((0,3), dtype=np.float32))
    hits_coords: np.ndarray = field(default_factory=lambda: np.zeros((0,3), dtype=np.float32))
    path_length: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    energies: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    phases: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    hit_obj_idx: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.int32))
    rays_energies_output: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    rays_phases_output: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    delay: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    medium_absorption_coeffs: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    medium_absorption_phases: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    absorption_coeffs: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    absorption_phases: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    reflection_coeffs: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    reflection_phases: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    refraction_coeffs: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    refraction_phases: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    scattering_coeffs: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    scattering_phases: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))

    # keyword-only function with mandatory recursion_idx and n_rays
    def add_data(self, recursion_idx: int, n_rays: int, origins: np.ndarray = None, directions: np.ndarray = None, hits_coords: np.ndarray = None, path_length: np.ndarray = None, delay: np.ndarray = None, energies: np.ndarray = None, phases: np.ndarray = None, medium_absorption_coeffs: np.ndarray = None, medium_absorption_phases: np.ndarray = None, absorption_coeffs: np.ndarray = None, absorption_phases: np.ndarray = None, reflection_coeffs: np.ndarray = None, reflection_phases: np.ndarray = None, refraction_coeffs: np.ndarray = None, refraction_phases: np.ndarray = None, scattering_coeffs: np.ndarray = None, scattering_phases: np.ndarray = None, output_mask: np.ndarray = None, intersect_mask: np.ndarray = None, rays_energies_output: np.ndarray = None, rays_phases_output: np.ndarray = None):

        # Check if recursion_idx is already present
        if not np.any(self.recursions == recursion_idx):
            self.recursions = np.append(self.recursions, np.full((n_rays,1), [recursion_idx], dtype=np.int32))

        if isinstance(origins, np.ndarray) and origins.shape[0] == n_rays:
            self.origins = np.append(self.origins, origins, axis=0).astype(np.float32)

        if isinstance(directions, np.ndarray) and directions.shape[0] == n_rays:
            self.directions = np.append(self.directions, directions, axis=0).astype(np.float32)

        if isinstance(hits_coords, np.ndarray) and hits_coords.shape[0] == n_rays:
            self.hits_coords = np.append(self.hits_coords, hits_coords, axis=0).astype(np.float32)

        if isinstance(path_length, np.ndarray) and path_length.shape[0] == n_rays:
            self.path_length = np.append(self.path_length, path_length, axis=0).astype(np.float32)

        if isinstance(delay, np.ndarray) and delay.shape[0] == n_rays:
            self.delay = np.append(self.delay, delay, axis=0).astype(np.float32)

        if isinstance(energies, np.ndarray) and energies.shape[0] == n_rays:
            self.energies = np.append(self.energies, energies, axis=0).astype(np.float32)

        if isinstance(phases, np.ndarray) and phases.shape[0] == n_rays:
            self.phases = np.append(self.phases, phases, axis=0).astype(np.float32)

        if isinstance(hit_obj_idx, np.ndarray) and hit_obj_idx.shape[0] == n_rays:
            self.hit_obj_idx = np.append(self.hit_obj_idx, hit_obj_idx, axis=0).astype(np.int32)

        if isinstance(rays_energies_output, np.ndarray) and rays_energies_output.shape[0] == n_rays:
            self.rays_energies_output = np.append(self.rays_energies_output, rays_energies_output, axis=0).astype(np.float32)

        if isinstance(rays_phases_output, np.ndarray) and rays_phases_output.shape[0] == n_rays:
            self.rays_phases_output = np.append(self.rays_phases_output, rays_phases_output, axis=0).astype(np.float32)
