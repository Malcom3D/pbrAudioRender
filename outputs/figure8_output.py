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
from dataclasses import dataclass

@dataclass
class Figure8Output:
    config: dict
    
    def capture_pressure(self, pressure: float, velocity: np.ndarray, position: tuple) -> float:
        """Capture pressure with figure-8 directivity pattern"""
        # Figure-8 pattern: cos(theta) where theta is angle between orientation and sound direction
        
        # For figure-8, we need the particle velocity direction
        if np.linalg.norm(velocity) == 0:
            return 0.0
            
        sound_direction = velocity / np.linalg.norm(velocity)
        orientation = self.get_orientation_vector()
        
        cos_theta = np.dot(orientation, sound_direction)
        return pressure * cos_theta
    
    def get_orientation_vector(self) -> np.ndarray:
        """Get microphone orientation vector"""
        rotation = self.config.get('rotation', (0.0, 0.0, 0.0))
        roll, pitch, yaw = rotation
        
        # Calculate rotation matrix (same as plane source)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R_y = np.array([
            [np.cos(pitch), 0,, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        R = R_z @ R_y @ R_x
        forward = np.array([1, 0, 0])
        return R @ forward
    
    @property
    def position(self):
        return self.config['position']
    
    @property
    def type(self):
        return "figure8"
