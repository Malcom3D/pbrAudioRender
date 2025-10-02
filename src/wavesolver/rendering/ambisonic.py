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
import soundfile as sf
from typing import List, Tuple, Optional, Dict
import math

class AmbisonicRenderer:
    """
    Render Ambisonic audio from pressure samples at output positions.
    Supports orders 0 to 3 with proper spherical harmonics.
    """
    
    def __init__(self, order: int = 1, sample_rate: int = 48000):
        self.order = min(max(order, 0), 3)  # Clamp to 0-3
        self.sample_rate = sample_rate
        self.channel_count = (self.order + 1) ** 2
        
        # Channel ordering for Ambisonic B-format
        self.channel_order = self._get_channel_order()
        
        # Precompute normalization factors
        self.normalization = self._compute_normalization_factors()
    
    def _get_channel_order(self) -> List[Tuple[int, int]]:
        """Get spherical harmonics channel order (order, degree)."""
        channel_order = []
        for order in range(self.order + 1):
            for degree in range(-order, order + 1):
                channel_order.append((order, degree))
        return channel_order
    
    def _compute_normalization_factors(self) -> Dict[Tuple[int, int], float]:
        """Compute spherical harmonics normalization factors (SN3D)."""
        normalization = {}
        
        for order, degree in self.channel_order:
            if order == 0:
                normalization[(order, degree)] = 1.0  # W channel
            else:
                # SN3D normalization
                kronecker = 1.0 if degree == 0 else 2.0
                factorial_ratio = math.factorial(order - abs(degree)) / math.factorial(order + abs(degree))
                normalization[(order, degree)] = math.sqrt(kronecker * factorial_ratio)
        
        return normalization
    
    def spherical_harmonic(self, order: int, degree: int, azimuth: float, elevation: float) -> float:
        """
        Compute real spherical harmonic value.
        
        Args:
            order: Spherical harmonic order
            degree: Spherical harmonic degree (-order to order)
            azimuth: Azimuth angle in radians (0 to 2π)
            elevation: Elevation angle in radians (-π/2 to π/2)
            
        Returns:
            Spherical harmonic value
        """
        # Convert to spherical coordinates (azimuth, inclination)
        inclination = math.pi / 2 - elevation  # Convert elevation to inclination
        
        if order == 0:
            # W channel (omnidirectional)
            return 1.0
            
        elif order == 1:
            # First order components
            if degree == -1:
                return math.sin(azimuth) * math.cos(inclination)  # Y
            elif degree == 0:
                return math.sin(inclination)                      # Z  
            elif degree == 1:
                return math.cos(azimuth) * math.cos(inclination)  # X
                
        elif order == 2:
            # Second order components
            if degree == -2:
                return math.sqrt(3) * math.sin(2 * azimuth) * math.cos(inclination) ** 2
            elif degree == -1:
                return math.sqrt(3) * math.sin(azimuth) * math.sin(2 * inclination)
            elif degree == 0:
                return (3 * math.sin(inclination) ** 2 - 1) / 2
            elif degree == 1:
                return math.sqrt(3) * math.cos(azimuth)) * math.sin(2 * inclination)
            elif degree == 2:
                return math.sqrt(3) * math.cos(2 * azimuth) * math.cos(inclination) ** 2
                
        elif order == 3:
            # Third order components
            sin_az = math.sin(azimuth)
            cos_az = math.cos(azimuth)
            sin_inc = math.sin(inclination)
            cos_inc = math.cos(inclination)
            
            if degree == -3:
                return math.sqrt(5) * sin_az * (3 - 4 * sin_inc**2) * cos_inc**2
            elif degree == -2:
                return math.sqrt(15) * sin_az * sin_inc * (1 - 4 * sin_inc**2) * cos_inc
            elif degree == -1:
                return math.sqrt(3) * sin_az * (2 - 5 * sin_inc**2) * sin_inc
            elif degree == 0:
                return sin_inc * (5 * sin_inc**2 - 3) / 2
            elif degree == 1:
                return math.sqrt(3) * cos_az * (2 - 5 * sin_inc**2) * sin_inc
            elif degree == 2:
                return math.sqrt(15) * cos_az * sin_inc * (1 - 4 * sin_inc**2) * cos_inc
            elif degree == 3:
:
                return math.sqrt(5) * cos_az * (3 - 4 * sin_inc**2) * cos_inc**2
        
        return 0.0
    
    def cartesian_to_spherical(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert Cartesian coordinates to spherical coordinates.
        
        Returns:
            Tuple of (distance, azimuth, elevation)
        """
        distance = math.sqrt(x**2 + y**2 + z**2)
        
        if distance == 0:
            return 0.0, 0.0, 0.0
        
        # Azimuth (0 to 2π)
        azimuth = math.atan2(y, x)
        if azimuth < 0:
            azimuth += 2 * math.pi
        
        # Elevation (-π/2 to π/2)
        elevation = math.asin(z / distance)
        
        return distance, azimuth, elevation
    
    def pressure_to_ambisonic(self, pressure_samples: List[float],
                             positions: List[Tuple[float, float, float]],
                             listener_position: Tuple[float, float, float],
                             distance_attenuation: bool = True) -> np.ndarray:
        """
        Convert pressure samples at multiple positions to Ambisonic B-format.
        
        Args:
            pressure_samples: List of pressure values at sample positions
            positions: List of (x, y, z) positions in world coordinates
            listener_position: Current listener position in world coordinates
            distance_attenuation: Whether to apply distance-based attenuation
            
        Returns:
            Ambisonic B-format audio frame
        """
        ambisonic_frame = np.zeros(self.channel_count, dtype=np.float32)
        
        for pressure, position in zip(pressure_samples, positions):
            # Convert to listener-relative coordinates
            rel_x = position[0] - listener_position[0]
            rel_y = position[1] - listener_position[1]
            rel_z = position[2] - listener_position[2]
            
            # Convert to spherical coordinates
            distance, azimuth, elevation = self.cartesian_to_spherical(rel_x, rel_y, rel_z)
            
            # Apply distance attenuation
            if distance_attenuation and distance > 0:
                # Inverse distance law with near-field correction
                attenuation = 1.0 / (1.0 + distance)
                effective_pressure = pressure * attenuation
            else:
                effective_pressure = pressure
            
            # Compute spherical harmonics contribution for each channel
            for channel_idx, (order, degree) in enumerate(self.channel_order):
                sh_value = self.spherical_harmonic(order, degree, azimuth, elevation)
)
                norm_factor = self.normalization.get((order, degree), 1.0)
                
                ambisonic_frame[channel_idx] += effective_pressure * sh_value * norm_factor
        
        return ambisonic_frame
    
    def render_animation(self, soxel_grid,, output_positions: List[Tuple[float, float, float]],
                        listener_trajectory: List[Tuple[float, float, float]],
                        output_file: str,
                        distance_attenuation: bool = True):
        """
        Render complete Ambisonic audio from simulation.
        
        Args:
            soxel_grid: SoxelGrid containing simulation data
            output_positions: List of positions to sample pressure from
            listener_trajectory: List of listener positions for each frame
            output_file: Output WAV file path
            distance_attenuation: Whether to apply distance-based attenuation
        """
        
        audio_frames = []
        
        for frame_idx in range(soxel_grid.num_frames):
            pressure_field, _ = soxel_grid.get_frame(frame_idx)
            listener_pos = listener_trajectory[frame_idx % len(listener_trajectory)]
            
            # Sample pressure at output positions
            pressure_samples = []
            for pos in output_positions:
                # Convert world position to grid coordinates
                grid_x = int(pos[0] / soxel_grid.voxel_size)
                grid_y = int(pos[1] / soxel_grid.voxel_size)
                grid_z = int(pos[2] / soxel_grid.voxel_size)
                
                # Clamp to grid boundaries
                grid_x = max(0, min(grid_x, soxel_grid.dimensions[0] - 1))
                grid_y = max(0, min(grid_y, soxel_grid.dimensions[1] - 1))
                grid_z = max(0, min(grid_z, soxel_grid.dimensions[2] - 1))
                
                pressure_samples.append(pressure_field[grid_x, grid_y, grid_z])
            
            # Convert to Ambisonic
            ambisonic_frame = self.pressure_to_ambisonic(
                pressure_samples, output_positions, listener_pos, distance_attenuation)
            audio_frames.append(ambisonic_frame)
        
        # Convert to numpy array
        audio_data = np.array(audio_frames, dtype=np.float32)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data /= max_val
        
        # Write to file
        sf.write(output_file, audio_data, self.sample_rate, subtype='FLOAT')
        
        print(f"Rendered {self.order} order Ambisonic audio to {output_file}")
        print(f"Channels: {self.channel_count}, Duration: {len(audio_data)/self.sample_rate:.2f}s")
    
    def get_channel_labels(self) -> List[str]:
        """Get labels for each Ambisonic channel."""
        labels = []
        for order, degree in self.channel_order:
            if order == 0:
                labels.append("W")
            elif order == 1:
                if degree == -1: labels.append("Y")
                elif degree == 0: labels.append("Z")
                elif degree == 1: labels.append("X")
            else:
                labels.append(f"_{order}{degree}")
        return labels
