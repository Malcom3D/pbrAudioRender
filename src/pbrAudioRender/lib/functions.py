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
import resampy
import soundfile as sf
import numba as nb
from typing import List, Tuple, Union, Dict, Any

from ..lib.filter import LinkwitzRileyFilter

def _append_to_npz(npz_path: str, value):
    """Append value to array saved as npz file """
    try:
        file_array = np.load(npz_path)
    except FileNotFoundError:
        array = np.array([value])
        np.savez_compressed(npz_path, array)
        return
    try:
        array = file_array[file_array.files[0]]
    except ValueError:
        file_array.allow_pickle = True
        array = file_array[file_array.files[0]]
    if type(value) == complex:
        array = np.append(array, [value])
    elif type(value) == list:
        array = np.append(array, [value], axis=0)
    np.savez_compressed(npz_path, array)
    file_array.close()
    
def _audio_to_npz(npz_path: str, audio_file: str, audio_npz: str, grid_sample_rate: int, frequency_bands: List[Tuple[float, float]]) -> str:
    """
    Convert audio_file to frequency dependent np.ndarray in npz file audio_npz.
    """
    # Read the audio file
    try:
        audio_data, sample_rate = sf.read(audio_file)
    except Exception as e:
        raise FileNotFoundError(f"Could not read audio file {audio_file}: {e}")

    # Ensure mono audio
    if audio_data.ndim > 1:
        if audio_data.shape[1] > 1:
            # Convert to mono by averaging channels
            audio_data = np.mean(audio_data, axis=1)
            print(f"Warning: Multi-channel audio converted to mono")

    # align sample_rate
    if sample_rate is grid_sample_rate:
        audio_data = resampy.resample(audio_data, sample_rate, grid_sample_rate)

    # create an array of np.ndarray for any frequency band
    multi_bands_audio = []
    for index in range(len(frequency_bands)-1):
        low_freq = frequency_bands[index][0]
        high_freq = frequency_bands[index][1]
        filtered_audio, sample_rate = LinkwitzRileyFilter.linkwitz_riley_bandpass_filter(audio_data, sample_rate, low_freq, high_freq)

        multi_bands_audio.append([low_freq, high_freq, filtered_audio])

    if not os.path.exists(npz_path):
        os.makedirs(npz_path)

    fd_samples = np.array(multi_bands_audio, dtype=object)

    audio_npz = os.path.join(npz_path, audio_npz)
    np.savez_compressed(audio_npz, fd_samples)
    return audio_npz

def _generate_band_frequencies(lowest_frequency: float, higher_frequency: float, steps_per_octave: int):
    """
    Generate frequencies from lowest_frequency to higher_frequency with specified steps per octave
    """
    frequencies = []
    current_freq = lowest_frequency

    # Calculate the frequency ratio for one step
    step_ratio = 2 ** (1 / steps_per_octave)

    while current_freq <= higher_frequency:
        frequencies.append(current_freq)
        current_freq *= step_ratio

    return frequencies

def _soxel_grid_shape(geometry: List[Tuple[float, float, float]], voxel_size: float) -> Tuple[int, int, int]:
    # compute acoustic domain shape from geometry
    geometry = geometry if isinstance(geometry, np.ndarray) else np.array(geometry)
    shape_z = (np.linalg.norm(geometry[0] - geometry[1]) / voxel_size).astype(int)
    shape_y = (np.linalg.norm(geometry[1] - geometry[2]) / voxel_size).astype(int)
    shape_x = (np.linalg.norm(geometry[2] - geometry[6]) / voxel_size).astype(int)
    return [int(shape_x), int(shape_y), int(shape_z)]

def _world_to_grid(voxel_size: float, grid_geometry: Union[list, np.ndarray], world_pos: Union[float, list, tuple, np.ndarray]) -> Tuple[int, int, int]:
    """Convert world coordinates to grid indices"""
    if isinstance(world_pos, float):
       return int(world_pos / voxel_size)
    grid_geometry = grid_geometry if isinstance(grid_geometry, np.ndarray) else np.array(grid_geometry)
    world_pos = world_pos if isinstance(world_pos, np.ndarray) else np.array(world_pos)
    grid_coords = ((world_pos - grid_geometry[0]) / voxel_size).astype(int)
    return grid_coords

def _is_in_bounds(shape: Tuple[int, int, int], i: int, j: int, k: int) -> bool:
    """Check if grid indices are within bounds"""
    return (0 < i < shape[0] and
            0 < j < shape[1] and
            0 < k < shape[2])

def _get_position(position_file: str, current_frame: int) -> Tuple[int, int, int]:
    """Get grid position at frame from npz file"""
    # Load position data from file if available
    if position_file and os.path.exists(position_file):
        try:
            centers = np.load(position_file)
            centers = centers[centers.files[0]]
            # Convert world coordinates to grid indices
            center = centers[current_frame]
    
            return center
        except Exception as e:
            print(f"Warnings: Failed to load position data from {position_file}: {e}")
    else:
        return np.array([0,0,0])

def _get_rotation(rotation_file: str, current_frame: int) -> Tuple[int, int, int]:
    """Get grid rotation at frame from npz file"""
    # Load rotation data from file if available
    if rotation_file and os.path.exists(position_file):
        try:
            directions = np.load(rotation_file)
            directions = directions[centers.files[0]]
            direction = directions[current_frame]
    
            return direction
        except Exception as e:
            print(f"Warnings: Failed to load direction data from {rotation_file}: {e}")
    else:
        return np.array([0,0,0])

def _cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert cartesian coordinates to spherical coordinates.

    Args:
        x, y, z: Cartesian coordinates

    Returns:
        (azimuth, elevation, radius) in degrees and units
    """
    radius = np.sqrt(x*x + y*y + z*z)

    if radius == 0:
        return 0.0, 0.0, 0.0

    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / radius)

    return azimuth, elevation, radius

def _degrees_to_radians(phase_coeffs, input_unit='auto'):
    """
    Verify if phase coefficients are in normalized radians and convert if needed.

    Parameters:
    -----------
    phase_coeffs : np.array
        Array of phase coefficients
    input_unit : str, optional
        Input unit type: 'radians', 'degrees', 'gradians', or 'auto' (default)
        If 'auto', the function will attempt to detect the unit

    Returns:
    --------
    normalized_phase : np.array
        normalized_phase : phase coefficients in normalized radians [-π, π]
    """

    # Make a copy to avoid modifying the original array
    phase = phase_coeffs.copy()
    was_normalized = False

    if input_unit == 'auto':
        # Auto-detection logic
        max_abs = np.max(np.abs(phase))
        
        if max_abs <= np.pi:
            # Likely already in radians
            original_unit = 'radians'
        elif max_abs <= 180:
            # Likely in degrees
            original_unit = 'degrees'
            was_normalized = True
        elif max_abs > 180:
            # Likely in gradians
            original_unit = 'gradians'
            was_normalized = True
        else:
            # Default to radians if uncertain
            original_unit = 'radians'
            print("Error: Could not auto-detect unit with certainty.")
            return None
    else:
        original_unit = input_unit
        was_normalized = (input_unit != 'radians')
   
    # Perform conversion if needed
    if original_unit == 'degrees':
        # Convert degrees to radians
        phase = np.deg2rad(phase)
        was_normalized = True
    elif original_unit == 'gradians':
        # Convert gradians to radians (200 gradians = 180 degrees = π radians)
        phase = phase * (np.pi / 200)
        was_normalized = True

    # Normalize to [-π, π] range
    if was_normalized or input_unit == 'radians':
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi

    return phase

@nb.jit(nopython=True)
def _trilinear_interpolate(field: np.ndarray, position: Tuple[float, float, float]) -> float:
    """Fast trilinear interpolation using numba"""
    i, j, k = position
    
    i0, j0, k0 = int(np.floor(i)), int(np.floor(j)), int(np.floor(k))
    i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
    
    # Check bounds
    if (i0 < 0 or i1 >= field.shape[0] or 
        j0 < 0 or j1 >= field.shape[1] or 
        k0 < 0 or k1 >= field.shape[2]):
        return 0.0
    
    # Calculate interpolation weights
    di, dj, dk = i - i0, j - j0, k - k0
    di1, dj1, dk1 = 1.0 - di, 1.0 - dj, 1.0 - dk

    # Get the 8 corner values
    v000 = field[i0, j0, k0]
    v001 = field[i0, j0, k1]
    v010 = field[i0, j1, k0]
    v011 = field[i0, j1, k1]
    v100 = field[i1, j0, k0]
    v101 = field[i1, j0, k1]
    v110 = field[i1, j1, k0]
    v111 = field[i1, j1, k1]

    # Trilinear interpolation
    c00 = v000 * di1 + v100 * di
    c01 = v001 * di1 + v101 * di
    c10 = v010 * di1 + v110 * di
    c11 = v011 * di1 + v111 * di

    c0 = c00 * dj1 + c10 * dj
    c1 = c01 * dj1 + c11 * dj
    
    value = c0 * dk1 + c1 * dk

    return value
