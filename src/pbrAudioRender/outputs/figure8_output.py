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
from typing import Dict, List, Tuple, Optional, Any
from ..lib.acoustic_io import AcousticEntity

class Figure8Output(AcousticEntity):
    """Figure-8 (bidirectional) microphone output"""
    
    def __init__(self, output_config):
        super().__init__(output_config, "output")
        self.output_config = output_config
    
    def get_recording_positions(self) -> List[Tuple[float, float, float]]:
        """Get positions where this output should record"""
        # Same implementation as other outputs
        positions = []
        
        if self.output_config.position_file:
            try:
                position_data = np.load(self.output_config.position_file)
                for pos in position_data:
                    if len(pos) >= 3:
                        positions.append(tuple(pos[:3]))
            except:
                print(f"Could not load positions from {self.output_config.position_file}")
        
        if not positions and 'position' in self.output_config.geometry:
            positions.append(tuple(self.output_config.geometry['position']))
        
        return positions
    
    def process_audio(self, pressure: float, velocity: np.ndarray,
                     position: Tuple[float, float, float], frame: int) -> float:
        """Process audio for figure-8 microphone"""
        # Figure-8 is proportional to velocity component
        mic_orientation = self._get_orientation(frame)
        velocity_component = np.dot(velocity, mic_orientation)
        
        sound_speed = 343.0
        density = 1.2
        figure8_signal = velocity_component / sound_speed * density
        
        # Apply frequency response and calibration
        processed = self.frequency_response.apply_response(
            np.array([figure8_signal]), frequency=1000
        )[0]
        
        if hasattr(self, 'calibration'):
            processed = self.calibration.apply_calibration(
                np.array([processed]), frequency=1000
            )[0]
        
        return processed
    
    def _get_orientation(self, frame: int) -> np.ndarray:
        """Get microphone orientation for current frame"""
        if self.output_config.position_file:
            try:
                position_data = np.load(self.output_config.position_file)
                if frame < len(position_data) and len(position_data[frame]) >= 7:
                    return position_data[frame][4:7]
            except:
                pass
        
        return np.array([1.0, 0.0, 00.0])
    
    def get_directivity(self, azimuth: float, elevation: float,
                       frequency: Optional[float] = None) -> float:
        """Figure-8 directivity pattern"""
        return np.cos(np.deg2rad(azimuth))

