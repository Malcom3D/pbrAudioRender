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
from typing import List, Dict
from numba import jit

class AmbisonicEncoder:
    def __init__(self, order: int = 3):
        self.order = order
        self.num_channels = (order + 1) ** 2
        
    def encode_output(self, output, output_path: str):
        """Encode A-format to B-format Ambisonics"""
        # # This would encode the captured A-format signals to B-format
        # For now, we'll create a placeholder implementation
        
        # In a complete implementation, this would:
        # 1. Load the captured A-format signals from the simulation
        # 2. Apply the appropriate encoding matrix based on the spatial arrangement
        # 3. Output B-format files (WAV) with ACN channel ordering and N3D normalization
        
        print(f"Encoding Ambisonic output to {output_path}")
        
    def a_format_to_b_format(self, a_format_signals: np.ndarray, 
                           output_positions: List) -> np.ndarray:
        """Convert A-format signals to B-format"""
        num_frames = a_format_signals.shape[0]
        num_a_channels = a_format_signals.shape[1]
        
        # Initialize B-format output
        b_format = np.zeros((num_frames, self.num_channels))
        
        # Calculate encoding matrix based on output positions
        encoding_matrix = self.calculate_encoding_matrix(output_positions)
        
        # Apply encoding matrix
        for frame in range(num_frames):
            b_format[frame] = encoding_matrix @ a_format_signals[frame]
            
        return b_format
    
    def calculate_encoding_matrix(self, positions: List) -> np.ndarray:
        """Calculate encoding matrix from output positions"""
        num_outputs = len(positions)
        encoding_matrix = np.zeros((self.num_channels, num_outputs))
        
        for i, pos in enumerate(positions):
            # Convert Cartesian to spherical coordinates
            x, y, z = pos
            r = np.sqrt(x**2 + y**2 + z**2)
            
            if r == 0:
                azimuth = 0
                elevation = 0
            else:
                azimuth = np.arctan2(y, x)
                elevation = np.arcsin(z / r)
            
            # Calculate spherical harmonics for each Ambisonic channel
            channel_idx = 0
            for n in range(self.order + 1):  # order
                for m in range(-n, n + 1):  # degree
                    encoding_matrix[channel_idx, i] = self.spherical_harmonic(
                        n, m, azimuth, elevation
                    )
                    channel_idx += 1
                    
        return encoding_matrix
    
    def spherical_harmonic(self, n: int, m: int, azimuth: float, elevation: float) -> float:
        """Calculate real spherical harmonic value"""
        # Normalization factor (N3D)
        normalization = np.sqrt((2 - (m == 0)) * np.math.factorial(n - abs(m)) / 
                              np.math.factorial(n + abs(m)))
        
        # Associated Legendre polynomial
        cos_theta = np.cos(elevation)
        sin_theta = np.sin(elevation)
        
        # This is a simplified implementation
        # A complete implementation would use proper Legendre polynomials
        
        if m == 0:
            return normalization * np.cos(n * azimuth)
        elif m > 0:
            return normalization * np.cos(m * azimuth) * np.cos(elevation)
        else:
            return normalization * np.sin(abs(m) * azimuth) * np.cos(elevation)
    
    def get_channel_names(self) -> List[str]:
        """Get ACN channel names for the current order"""
        channel_names = []
        for n in range(self.order + 1):
            for m in range(-n, n + 1):
                if n == 0:
                    channel_names.append("W")
                elif n == 1:
                    if m == -1: channel_names.append("Y")
                    elif m == 0: channel_names.append("Z") 
                    elif m == 1: channel_names.append("X")
                else:
                    channel_names.append(f"_{n}{m}")
        return channel_names
