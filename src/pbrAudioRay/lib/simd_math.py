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
from numba import float64, int32

@nb.njit(fastmath=True, cache=True)
def dot_product_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
       """Batch dot product using SIMD"""
    n = a.shape[0]
    result = np.zeros(n, dtype=np.float64)
    for i in nb.prange(n):
        result[i] = a[i, 0]*b[i, 0] + a[i, 1]*b[i, 1] + a[i, 2]*b[i, 2]
    return result

@nb.njit(fastmath=True, cache=True)
def cross_product_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batch cross product using SIMD"""
    n = a.shape[0]
    result = np.zeros((n, 3), dtype=np.float64)
    for i in nb.prange(n):
        result[i, 0] = a[i, 1]*b[i, 2] - a[i, 2]*b[i, 1]
        result[i, 1] = a[i, 2]*b[i, 0] - a[i, 0]*b[i, 2]
        result[i, 2] = a[i, 0]*b[i, 1] - a[i, 1]*b[i, 0]
    return result

@nb.njit(fastmath=True, cache=True)
def normalize_batch(vectors: np.ndarray) -> np.ndarray:
    """Batch vector normalization using SIMD"""
    n = vectors.shape[0]
    result = np.zeros_like(vectors)
    for i in nb.prange(n):
        norm = np.sqrt(vectors[i, 0]**2 + vectors[i, 1]**2 + vectors[i, 2]**2)
        if norm > 1e-12:
            result[i] = vectors[i] / norm
        else:
            result[i] = vectors[i]
    return result

