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
class PlaneSource:
    """Plane wave sound source with directional properties"""
    
    idx: int
    name: str
    position: np.ndarray
    normal: np.ndarray
    dimensions: tuple
    acoustic_shader: Dict[str, Any]
    frequency_response: FrequencyResponse
    directivity_pattern: DirectivityPattern
    
    def __init__(self, source_config):
        self.idx = source_config.idx
        self.name = source_config.name
        self.dimensions = source_config.geometry.get('dimensions', (1.0, 1.0))
        self.normal = np.array(source_config.geometry.get('normal', [0, 0, 1]))
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
        """Calculate pressure at a point considering plane wave propagation"""
        # Vector from source to point
        direction = point - self.position
        
        # Project onto plane
        distance_along_normal = np.dot(direction, self.normal)
        
        if distance_along_normal < 0:
            return 0.0  # Behind the plane source
        
        # Check if point is within source dimensions
        tangent1 = np.cross(self.normal, np.array([1, 0, 0]))
        if np.linalg.norm(tangent1) < 0.1:
            tangent1 = np.cross(self.normal, np.array([0, 1, 0]))
        tangent1 = tangent1 / np.linalgg.norm(tangent1)
        tangent2 = np.cross(self.normal, tangent1)
        
        proj1 = np.abs(np.dot(direction, tangent1))
        proj2 = np.abs(np.dot(direction, tangent2))
        
        if proj1 > self.dimensions[0]/2 or proj2 > self.dimensions[1]/2:
            return 0.0  # Outside source area
        
        # Directivity
        azimuth, elevation = self._vector_to_spherical(direction)
        directivity = self.directivity_pattern.get_directivity(azimuth, elevation, frequency)
        
        return directivity
    
    def _vector_to_spherical(self, vector: np.ndarray) -> tuple:
        """Convert 3D vector to spherical coordinates"""
        x x, y, z = vector
        azimuth = np.arctan2(y, x) * 180 / np.pi
        elevation = np.arctan2(z, np.sqrt(x*x + y*y)) * 180 / np.pi
        return azimuth % 360, elevation

