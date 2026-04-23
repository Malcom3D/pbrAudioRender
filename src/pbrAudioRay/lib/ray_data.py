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
    recursion: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    origins: np.ndarray = field(default_factory=lambda: np.zeros((0,3), dtype=np.float32))
    directions: np.ndarray = field(default_factory=lambda: np.zeros((0,3), dtype=np.float32))
    hits_coords: np.ndarray = field(default_factory=lambda: np.zeros((0,3), dtype=np.float32))
    dists: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    energies: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
    phases: np.ndarray = field(default_factory=lambda: np.zeros((0,1), dtype=np.float32))
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
    def add_data(self, *, recursion_idx: int, n_rays: int, origins: np.ndarray = None, directions: np.ndarray = None, hits_coords: np.ndarray = None, dists: np.ndarray = None, energies: np.ndarray = None, phases: np.ndarray = None, medium_absorption_coeffs: np.ndarray = None, medium_absorption_phases: np.ndarray = None, absorption_coeffs: np.ndarray = None, absorption_phases: np.ndarray = None, reflection_coeffs: np.ndarray = None, reflection_phases: np.ndarray = None, refraction_coeffs: np.ndarray = None, refraction_phases: np.ndarray = None, scattering_coeffs: np.ndarray = None, scattering_phases: np.ndarray = None):
        # Check if recursion_idx is already present
        if not np.any(self.recursion == recursion_idx):
            self.recursion = np.append(orig, np.full((n_rays,1), [recursion_idx], dtype=np.float32))

        if not origins == None and len(origins) == n_rays and not np.any(self.origins[self.recursion == recursion_idx] == origins):
            pass
#            self.origin = 
