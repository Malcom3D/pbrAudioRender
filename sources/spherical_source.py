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
from typing import Dict, Any
from dataclasses import dataclass
from lib.directivity_pattern import DirectivityPattern
from lib.frequency_response import FrequencyResponse

@dataclass
class SphericalSource:
    """Spherical sound source with frequency-dependent properties"""
    
    idx: int
    name: str
    position: np.ndarray
    radius: float
    acoustic_shader: Dict[str, Any]
    frequency_response: FrequencyResponse
    directivity_pattern: DirectivityPattern
    
    def __init__(self, source_config):
        self.idx = source_config.idx
        self.name = source_config.name
        self.radius = source_config.geometry.get('radius', 0.1)
        self.acoustic_shader = source_config.acoustic_shader
        
        # Initialize frequency response
        if source_config.frequency_response:
            self.frequency_response = FrequencyResponse(source_config.frequency_response)
        elif source_config.frequency_response_file:
            self.frequency_response = FrequencyResponse()
            self.frequency_response.load_from_file(source_config.frequency_response_file)
        else:
            self.frequency_response = FrequencyResponse()
        
        # Initialize directivity pattern
        if source_config.directivity_pattern:
            self.directivity_pattern = DirectivityPattern(source_config.directivity_pattern)
        elif source_config.directivity_pattern_file:
            self.directivity_pattern = DirectivityPattern()
            self.directivity_pattern.load_from_file(source_config.directivity_pattern_file)
        else:
            self.directivity_pattern = DirectivityPattern()
    
    def get_pressure_at_point(self, point: np.ndarray, frequency: float) -> float:
        """Calculate pressure at a point considering spherical wave propagation"""
        distance = np.linalg.norm(point - self.position)
        
        if distance < self.radius:
            return 1.0  # Inside source
        
        # Spherical wave attenuation
        attenuation = self.radius / distance
        
        # Directivity
        direction = point - self.position
        azimuth, elevation = self._vector_to_spherical(direction)
        directivity = self.directivity_pattern.get_directivity(azimuth, elevation, frequency)
        
        return attenuation * directivity
    
    def _vector_to_spherical(self, vector: np.ndarray) -> tuple:
        """Convert 3D vector to spherical coordinates"""
        x, y, z = vector
        azimuth = np.arctan2(y, x) * 180 / np.pi
        elevation = np.arctan2(z, np.sqrt(x*x + y*y)) * 180 / np.pi
        return azimuth % 360, elevation

