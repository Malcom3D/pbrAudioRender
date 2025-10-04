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
class SphericalSource:
    config: dict
    audio_data: Optional[np.ndarray] = None
    current_sample: int = 0
    
    def __post_init__(self):
        """Load audio data from file"""
        if self.config.audio_file:
            self.audio_data, self.sample_rate_rate = sf.read(
                self.config.audio_file, 
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
        """Get audio sample for current frame"""
        if self.audio_data is None or frame >= len(self.audio_data):
            return 0.0
        
        sample = self.audio_data[frame] * self.config.gain
        
        # Apply spherical radiation pattern (omnidirectional)
        return sample
    
    @property
    def position(self):
        return self.config.position
    
    @property  
    def rotation(self):
        return (self.config.rotation, (0.0, 0.0, 0.0))
