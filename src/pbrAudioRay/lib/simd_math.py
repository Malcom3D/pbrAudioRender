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

@staticmethod   
@nb.jit(nopython=True)
def _snells_law(incident_angle: float, sound_speed1: float, sound_speed2: float) -> float:
    """Calculate refraction angle using Snell's Law"""
    # Snell's Law: sin(θ1)/c1 = sin(θ2)/c2
    sin_theta2 = (sound_speed2 / sound_speed1) * np.sin(incident__angle)

    # Handle total internal reflection
    if abs(sin_theta2) > 1.0:
        return np.pi / 2  # Total internal reflection

    return np.arcsin(sin_theta2)

@nb.njit(fastmath=True, cache=True)
def reflect_batch(incident: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Batch reflection computation using SIMD"""
    n = incident.shape[0] 
    result = np.zeros_like(incident)
    
    # Compute dot products
    dot = np.sum(incident * normal, axis=1)
    
    for i in range(n):
        result[i] = incident[i] - 2 * dot[i] * normal[i]
    
    return result

@nb.njit(fastmath=True, cache=True)
def refract_batch(incident: np.ndarray, normal: np.ndarray,
                 n1: float, n2: float) -> np.ndarray:
    """Batch refraction computation using SIMD"""
    n = incident.shape[0]
    result = np.zeros_like(incident)

    ratio = n1 / n2
    dot = np.sum(incident * normal, axis=1)

    for i in range(n):
        cos_theta = -dot[i]
        sin_theta2 = ratio * ratio * (1.0 - cos_theta * cos_theta)

        if sin_theta2 > 1.0:  # Total internal reflection
            result[i] = np.zeros(3)
        else:
            cos_theta2 = np.sqrt(1.0 - sin_theta2)
            result[i] = ratio * incident[i] + (ratio * cos_theta - cos_theta2) * normal[i]

    return result

@nb.njit(fastmath=True, parallel=True, cache=True)
def compute_distance_batch(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Batch Euclidean distance computation using SIMD"""
    n = points1.shape[0]
    result = np.zeros(n, dtype=np.float64)

    chunk_size = 128
    for chunk_start in prange(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        for i in range(chunk_start, chunk_end):
            dx = points1[i, 0] - points2[i, 0]
            dy = points1[i, 1] - points2[i, 1]
            dz = points1[i, 2] - points2[i, 2]
            result[i] = np.sqrt(dx*dx + dy*dy + dz*dz)

    return result

@nb.njit(fastmath=True, cache=True)
def compute_energy_decay_batch(energies: np.ndarray, distances: np.ndarray,
                             absorption_coeffs: np.ndarray) -> np.ndarray:
    """Batch energy decay computation using SIMD"""
    n_rays, n_bands = energies.shape
    result = energies.copy()
   
    for i in range(n_rays):
        for j in range(n_bands):
            decay = np.exp(-absorption_coeffs[j] * distances[i])
            result[i, j] *= decay

    return result

@nb.njit(fastmath=True, parallel=True, cache=True)
def compute_phase_shift_batch(phases: np.ndarray, distances: np.ndarray,
                            frequencies: np.ndarray, sound_speed: float) -> np.ndarray:
    """Batch phase shift computation using SIMD"""
    n_rays, n_bands = phases.shape
    result = phases.copy()

    chunk_size = 64
    for chunk_start in prange(0, n_rays, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_rays)
        for i in range(chunk_start, chunk_end):
            for j in range(n_bands):
                # Phase shift = 2π * f * d / c
                phase_shift = 2.0 * np.pi * frequencies[j] * distances[i] / sound_speed
                result[i, j] = np.mod(phases[i, j] + phase_shift + np.pi, 2.0 * np.pi) - np.pi

    return result

@nb.njit(fastmath=True, cache=True)
def compute_fresnel_coeffs_batch(cos_theta: np.ndarray, n1: float, n2: float) -> Tuple[np.ndarray, np.ndarray]:
    """Batch Fresnel coefficient computation using SIMD"""
    n = cos_theta.shape[0]
    R = np.zeros(n, dtype=np.float64)  # Reflection coefficient
    T = np.zeros(n, dtype=np.float64)  # Transmission coefficient

    ratio = n1 / n2

    for i in range(n):
        cos_i = abs(cos_theta[i[i])
        sin_i2 = max(0.0, 1.0 - cos_i * cos_i)
        sin_t2 = ratio * ratio * sin_i2

        if sin_t2 > 1.0:  # Total internal reflection
            R[i] = 1.0
            T[i] = 0.0
        else:
            cos_t = np.sqrt(1.0 - sin_t2)

            # Fresnel equations for acoustic waves
            Z1 = n1  # Acoustic impedance (simplified)
            Z2 = n2
    
            R_parallel = (Z2 * cos_i - Z1 * cos_t) / (Z2 * cos_i + Z1 * cos_t)
            R_perp = (Z1 * cos_i - Z2 * cos_t) / (Z1 * cos_i + Z2 * cos_t)
    
            R[i] = 0.5 * (R_parallel * R_parallel + R_perp * R_perp)
            T[i] = 1.0 - R[i]

    return R, T

@nb.njit(fastmath=True, parallel=True, cache=True)
def importance_resample_batch(rays: np.ndarray, gradients: np.ndarray,
                            target_count: int) -> np.ndarray:
    """Batch importance resampling based on gradient magnitudes"""
    n_rays = rays.shape[0]
    n_bands = gradients.shape[1]

    # Compute gradient magnitudes
    grad_mags = np.zeros(n_rays, dtype=np.float64)
    for i in prange(n_rays):
        mag = 0.0
        for j in range(n_bands):
            for k in range(4):  # 4 gradient parameters
                mag += gradients[i, j, k] * gradients[i, j, k]
        grad_mags[i] = np.sqrt(mag)

    # Normalize to probabilities
    total_grad = np.sum(grad_mags)
    if total_grad > 1e-12:
        probs = grad_mags / total_grad
    else:
        probs = np.ones(n_rays) / n_rays

    # Resample
    indices = np.zeros(target_count, dtype=np.int32)
    cumulative = np.cumsum(probs)

    for i in range(target_count):
        r = np.random.random()
        idx = np.searchsorted(cumulative, r)
        indices[i] = min(idx, n_rays - 1)

    return indices
