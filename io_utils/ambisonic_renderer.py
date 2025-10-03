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
from typing import List, Tuple
#import ambix as am

class AmbisonicRenderer:
    """Render ambisonic output from pressure fields"""
    
    def __init__(self, order: int = 3, sample_rate: int = 48000):
        self.order = order
        self.sample_rate = sample_rate
        self.num_channels = (order + 1) ** 2
        
#    def pressure_to_ambisonics(self, pressures: List[Tuple[float, float, float, float]], 
#                              listener_pos: Tuple[float, float, float]) -> np.ndarray:
    def pressure_to_ambisonics(self, pressures: np.ndarray, 
                              listener_pos: Tuple[float, float, float]) -> np.ndarray:
        """Convert pressure samples at positions to ambisonic B-format"""
        # pressures: list of (x, y, z, pressure)
        ambisonic_frame = np.zeros(self.num_channels)
        
        for x, y, z, pressure in pressures:
            # Convert to spherical coordinates relative to listener
            rel_x = x - listener_pos[0]
            rel_y = y - listener_pos[1]
            rel_z = z - listener_pos[2]
            
            # Calculate spherical spherical harmonics coefficients
            r = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
            if r == 0:
                continue
                
            theta = np.arctan2(rel_y, rel_x)  # azimuth
            phi = np.arccos(rel_z / r)        # elevation
            
            # Compute spherical harmonics up to desired order
            channel_idx = 0
            for n in range(self.order + 1):
                for m in range(-n, n + 1):
                    Y = self.spherical_harmonic(n, m, theta, phi)
                    ambisonic_frame[channel_idx] += pressure * Y
                    channel_idx += 1
        
        return ambisonic_frame
    
    @staticmethod
    def spherical_harmonic(n: int, m: int, theta: float, phi: float) -> float:
        """Compute real spherical harmonic"""
        from scipy.special import sph_harm
        
        # Complex spherical harmonic
        Y_complex = sph_harm(abs(m), n, theta, phi)
        
        # Convert to real form
        if m == 0:
            return Y_complex.real
        elif m > 0:
            return np.sqrt(2) * (-1)**m * Y_complex.real
        else:
            return np.sqrt(2) * (-1)**m * Y_complex.imag
