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
from dataclasses import dataclass

@dataclass
class AcousticRay:
    """Ray data structure for multiple frequency bands with SIMD optimization"""
    n_rays: int
    n_freq_bands: int
    max_depth: int = 10

    def __post_init__(self):
        """
        Initialize ray data structure for vectorized operations.

        Parameters:
        -----------
        n_rays : int
            Number of rays to trace simultaneously
        n_freq_bands : int
            Number of frequency bands (typically 1-8 for acoustic simulation)
        max_depth : int
            Maximum ray recursion depth
        """
        n_rays = self.n_rays
        n_freq_bands = self.n_freq_bands
        max_depth = self.max_depth

        # Ray origin and direction (SIMD-friendly layout)
        self.origins = np.zeros((n_rays, 3), dtype=np.float32)
        self.directions = np.zeros((n_rays, 3), dtype=np.float32)

        # Frequency-dependent energy (stored as complex for phase information)
        self.energy = np.zeros((n_rays, n_freq_bands), dtype=np.complex64)

        # Ray state flags
        self.active = np.ones(n_rays, dtype=np.bool_)
        self.depth = np.zeros(n_rays, dtype=np.int32)

        # Intersection results
        self.hits = np.zeros(n_rays, dtype=np.bool_)
        self.distances = np.full(n_rays, np.inf, dtype=np.float32)
        self.face_ids = np.full(n_rays, -1, dtype=np.int32)
        self.barycentric = np.zeros((n_rays, 3), dtype=np.float32)

        # Material properties at intersection
        self.absorption = np.zeros((n_rays, n_freq_bands), dtype=np.float32)
        self.reflection = np.zeros((n_rays, n_freq_bands), dtype=np.float32)
        self.refraction = np.zeros((n_rays, n_freq_bands), dtype=np.float32)
        self.scattering = np.zeros((n_rays, n_freq_bands), dtype=np.float32)
        self.diffraction = np.zeros((n_rays, n_freq_bands), dtype=np.float32)
        self.normals = np.zeros((n_rays, 3), dtype=np.float32)

        # Path information for differentiable tracing
        self.path_lengths = np.zeros((n_rays, max_depth), dtype=np.float32)
        self.path_vertices = np.zeros((n_rays, max_depth, 3), dtype=np.float32)
        self.path_materials = np.zeros((n_rays, max_depth, n_freq_bands, 5), dtype=np.float32)  # [absorption, reflection, refraction, scattering, diffraction]

    @staticmethod
    @nb.njit(nogil=True, fastmath=True, cache=True)
    def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """Normalize array of vectors using SIMD-friendly operations"""
        norms = np.sqrt(np.sum(vectors**2, axis=1))
        norms[norms == 0] = 1.0  # Avoid division by zero
        return vectors / norms[:, np.newaxis]

    @staticmethod
    @nb.njit(nogil=True, fastmath=True, cache=True)
    def reflect_rays(directions: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Reflect ray directions using vectorized operations"""
        # directions and normals are (n, 3) arrays
        dot = np.sum(directions * normals, axis=1)
        return directions - 2 * dot[:, np.newaxis] * normals
