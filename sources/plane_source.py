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
from dataclasses import dataclass
from typing import Optional

@dataclass
class PlaneSource:
    config: dict
    audio_data: Optional[np.ndarray] = None
    current_sample: int = 0
    
    def __post_init__(self):
        """Load audio data from file"""
        if self.config.get('audio_file'):
            self.audio_data, self.sample_rate = sf.read(
                self.config['audio_file'], 
                dtype='float32',
                always_2d=False
            )
            # Convert to mono if needed
            if len(self.audio_data.shape) > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)
        else:
            # Generate test signal if no file provided
            duration = 1.0  # seconds
            t = np.linspace(0, duration, int(self.config.get('sample_rate', 48000) * duration))
            self.audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    def get_sample(self, frame: int) -> float:
        """Get audio sample for current frame with plane wave directivity"""
        if self.audio_data is None or frame >= len(self.audio_data):
            return 0.0
        
        sample = self.audio_data[frame] * self.config.get('gain', 1.0)
        
        # Plane waves have directional characteristics
        # For now, we'll implement basic plane wave behavior
        # More sophisticated directivity can be added later
        return sample
    
    def get_directivity(self, listener_pos: tuple) -> float:
        """Calculate directivity factor based on listener position"""
        source_pos = np.array(self.position)
        listener_pos = np.array(listener_pos)
        
        # Vector from source to listener
        direction = listener_pos - source_pos
        direction_normalized = direction / np.linalg.norm(direction)
        
        # Source orientation
        orientation = self.get_orientation_vector()
        
        # Cosine of angle between source orientation and listener direction
        cos_theta = np.dot(orientation, direction_normalized)
        
        # Plane wave directivity (cardioid pattern)
        directivity = 0.5 * (1 + cos_theta)
        
        return directivityivity
    
    def get_orientation_vector(self) -> np.ndarray:
        """Convert Euler angles to orientation vector"""
        roll, pitch, yaw = self.rotation
        
        # Calculate rotation matrix
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos.cos(roll)]
        ])
        
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix
        R = R_z @ R_y @ R_x
        
        # Default forward vector
        forward = np.array([1, 0, 0])
        
        # Rotated forward vector
        orientation = R @ forward
        
        return orientation
    
    @property
    def position(self):
        return self.config['position']
    
    @property  
    def rotation(self):
        return self.config.get('rotation', (0.0, 0.0, 0.0))
