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
from typing import Tuple, Optional
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.functions import _audio_to_npz, _get_position, _get_rotation, _world_to_grid, _is_in_bounds, _cartesian_to_spherical
from ..lib.acoustic_field import VelocityVectors, AcousticField
from ..lib.soxel import Soxel

@dataclass
class SphericalSource:
    """
    Spherical source
    """
    entity_manager: EntityManager
    idx: int

    def __post_init__(self):
        """
        Convert audio_file for source.idx to frequency dependent np.ndarray in npz file.
        """
        config = self.entity_manager.get('config')
        frequency_bands = self.entity_manager.get('frequency_bands')
        bands = frequency_bands.get_bands()
        self.frames = self.entity_manager.get('frames')
        self.shape = config.acoustic_domain.shape
        self.voxel_size = config.acoustic_domain.voxel_size
        self.grid_geometry = config.acoustic_domain.geometry
        self.sound_speed = config.acoustic_domain.acoustic_shader.sound_speed
        self.density = config.acoustic_domain.acoustic_shader.density
        grid_sample_rate = config.acoustic_domain.sample_rate

        for source_config in config.sources:
            if source_config.idx == self.idx:
                self.source_config = source_config

        audio_file = config.sources[self.idx].audio_file
        npz_path = os.path.join(config.system.cache_path, 'filtered_audio')
        audio_npz = str(self.idx) + '.npz'

        audio_npz = _audio_to_npz(npz_path, audio_file, audio_npz, grid_sample_rate, bands)

        # Open for get_field function
        fd_samples = np.load(audio_npz)
        fd_samples.allow_pickle = True
        self.fd_samples = fd_samples[fd_samples.files[0]]

    def get_soxels(self):
        """Voxelize a spherical sound source into the grid"""

        current_frame = self.frames.get()

        center_pos = _get_position(self.source_config.position_file, current_frame)
        center = _world_to_grid(self.voxel_size, self.grid_geometry, center_pos)
        radius = _world_to_grid(self.voxel_size, self.grid_geometry, self.source_config.geometry)
        voxels = self._get_sphere_voxels(center, radius)

        # Update soxels at source positions
        soxels = []
        for voxel in voxels:
            i, j, k = voxel
            x, y, z = center
            if _is_in_bounds(self.shape, i, j, k):
                dist = np.linalg.norm(center - voxel)
                if radius >= dist > radius-1:
                    type = 1  # Mark as source
                    input_pressures = self.get_field(center=center, point=voxel, sound_speed=self.sound_speed, density=self.density)
                else:
                    type = 2  # Mark as object - acoustic_shader
                    input_pressures = None

                soxel = Soxel(
                    idx=self.source_config.idx,
                    type = type,
                    input_pressures = input_pressures,
                    acoustic_shader = self.source_config.acoustic_shader
                )
                soxels.append([i,j,k,soxel])
        return soxels

    def _get_sphere_voxels(self, center: Tuple[int, int, int], radius: Tuple[int, int, int]):
        """
        Optimized version using distance squared to avoid sqrt calculations.
        """
        # Calculate bounding box
        min_x, min_y, min_z = center-radius
        max_x, max_y, max_z = center+radius
        
        voxels = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                for z in range(min_z, max_z + 1):
                    dist = np.linalg.norm(np.array(center) - np.array([x,y,z]))
                    if dist <= radius:
                        voxels.append([x, y, z])
        return voxels

    def update(self):
        """Update soxel grid for current frame"""

        current_frame = self.frames.get()

        # Initialize the grid
        #self._initialize_grid()

    def get_field(self, center: Tuple[int, int, int], point: Tuple[int, int, int], sound_speed: float = None, density: float = None) -> AcousticField:
        """
        Compute acoustic pressure and velocity vectors at the boundary of a spherical sound source.

        Parameters:
        -----------
        center: Tuple[int, int, int]
            Center of sphere
        point: Tuple[int, int, int]
            Coordinate of point on sphere surface
        sound_speed : float
            Sound speed of medium on boundary
        density : float
            Density of medium on boundary 

        Returns:
        --------
        AcousticField: AcousticField
        """

        current_frame = self.frames.get()

        if sound_speed == None:
           sound_speed = self.sound_speed
        if density == None:
           density = self.density

        # carthesian to azimuth and elevation
        x, y, z = np.array(point) - np.array(center)
        azimuth, elevation, r = _cartesian_to_spherical(x,y,z)

        fd_field = AcousticField([])
        for index in range(len(self.fd_samples)):
            low_freq = self.fd_samples[index][0]
            high_freq = self.fd_samples[index][1]
            sample = self.fd_samples[index][2][current_frame]

            magnitude_coeff = self.source_config.spatial_freq_response.get_avg_magnitude(azimuth, elevation, low_freq, high_freq)
            phase_coeff = self.source_config.spatial_freq_response.get_avg_phase(azimuth, elevation, low_freq, high_freq)

            # Apply both magnitude and phase coefficients
            # Treat the sample as a complex phasor for proper phase manipulation
            complex_sample = sample * magnitude_coeff * np.exp(1j * phase_coeff)
        
            # Use real part for acoustic pressure calculation
            sample = np.real(complex_sample)

            pressure, velocity = self._compute_acoustic_fields(azimuth, elevation, low_freq, high_freq, sample, self.sound_speed, self.density)
            velocity_vectors = VelocityVectors(velocity[0], velocity[1], velocity[2])
            fd_field.add_field(low_freq=low_freq, high_freq=high_freq, pressure=pressure, velocity=velocity_vectors)
        return fd_field

    def _compute_acoustic_fields(self, azimuth: float, elevation: float, low_freq: float, high_freq: float, sample: float, sound_speed: float, density: float) -> Tuple[float, np.ndarray]:
        """
        Compute acoustic pressure and velocity vectors at the boundary of a spherical sound source.
    
        Parameters:
        -----------
        low_freq : float
            Lower frequency bound for filtering (Hz)
        high_freq : float
            Upper frequency bound for filtering (Hz)
        sample : float
            Spatialtial Frequency Responce aware sound sample
        sound_speed : float
            Speed of sound (m/s), default 343.0
        density : float
            Density (kg/m³), default 1.225
    
        Returns:
        --------
        pressure : np.ndarray
            Acoustic pressure at boundary (Pa)
        velocity_vectors : np.ndarray
            Velocity vectors [(vx, vy, vz)] (m/s)
        """
        radius = self.source_config.geometry
    
        # Compute acoustic pressure
        if radius is None:
            # Point source approximation
            pressure = sample
        else:
            # Spherical source with radius
            # For a pulsating sphere, pressure is related to surface velocity
            # This is a simplified model - you might want to use more sophisticated acoustic models
            k = 2 * np.pi * np.sqrt(low_freq * high_freq) / sound_speed  # Wave number (approximate)
        
            # For a pulsating sphere, the pressure at surface is:
            # p = (ρc * v) / (1 + 1/(jkr)) where v is surface velocity
            # We'll use a simplified version for this example
            surface_velocity = sample
        
            # Complex impedance approach for spherical wave
            kr = k * radius
            if kr < 0.1:  # Small sphere approximation
                pressure = density * sound_speed * surface_velocity
            else:
                # More general spherical wave solution
                Z = density * sound_speed * (1j * kr) / (1 + 1j * kr)
                pressure = np.real(Z * surface_velocity)
    
        # Compute velocity vectors
        # For a spherical source, velocity is radial and proportional to pressure gradient
        velocity_magnitude = np.abs(pressure) / (density * sound_speed)
    
        # Create velocity vectors in 3D space
        # For a spherical source, velocity is radial from center
        # We'll create unit vectors pointing outward
        # For demonstration, we'll assume the source is at origin and compute
        # velocity vectors for points on a unit sphere
    
        v_mag = velocity_magnitude
        vx = velocity_magnitude * np.cos(elevation) * np.cos(azimuth)
        vy = velocity_magnitude * np.cos(elevation) * np.sin(azimuth)
        vz = velocity_magnitude * np.sin(elevation)

        velocity_vectors = np.array([vx, vy, vz])

        return pressure, velocity_vectors
