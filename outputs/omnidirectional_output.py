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

"""
Omnidirectional microphone output with uniform directivity.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from ..lib.directivity_pattern import DirectivityPattern, CommonDirectivityPatterns
from ..lib.frequency_response import FrequencyResponse
from ..lib.calibration import OutputCalibration


@dataclass
class OmnidirectionalOutput:
    """Omnidirectional microphone with uniform directivity in all directions"""
    
    idx: int
    name: str
    position: np.ndarray
    orientation: np.ndarray  # Quaternion or Euler angles
    frequency_response: FrequencyResponse
    directivity_pattern: DirectivityPattern
    calibration: OutputCalibration
    
    def __init__(self, output_config):
        self.idx = output_config.idx
        self.name = output_config.name
        self.position = np.array([0.0, 0.0, 0.0])  # Will be updated from position file
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Default quaternion
        
        # Initialize frequency response
        if output_config.frequency_response:
            self.frequency_response = FrequencyResponse(output_config.frequency_response)
        elif output_config.frequency_response_file:
            self.frequency_response = FrequencyResponse()
            self.frequency_response.load_from_file(output_config.frequency_response_file)
        else:
            self.frequency_response = FrequencyResponse()
        
        # Initialize directivity pattern (omnidirectional)
        if output_config.directivity_pattern:
            self.directivity_pattern = DirectivityPattern(output_config.directivity_pattern)
        elif output_config.directivity_pattern_file:
            self.directivity_pattern = DirectivityPattern()
            self.directivity_pattern.load_from_file(output_config.directivity_pattern_file)
        else:
            # Default omnidirectional pattern
            self.directivity_pattern = CommonDirectivityPatterns.omnidirectional()
        
        # Initialize calibration
        self.calibration = OutputCalibration()
    
    def get_sensitivity(self, frequency: float = 1000.0) -> float:
        """Get microphone sensitivity at specific frequency"""
        cal_params = self.calibration.get_calibration_at_frequency(frequency)
        return cal_params['total_gain_db']
    
    def get_directivity(self, azimuth: float, elevation: float, 
                       frequency: Optional[float] = None) -> float:
        """
        Get directivity coefficient for given direction.
        
        Args:
            azimuth: Azimuth angle in degrees (0-360)
            elevation: Elevation angle in degrees (-90 to 90)
            frequency: Frequency in Hz (optional)
        
        Returns:
            Directivity coefficient (linear scale)
        """
        return self.directivity_pattern.get_directivity(azimuth, elevation, frequency)
    
    def record_pressure(self, pressure: float, source_direction: Tuple[float, float],
                       frequency: Optional[float] = None) -> float:
        """
        Record pressure with directivity and frequency response applied.
        
        Args:
            pressure: Incident pressure
            source_direction: Tuple of (azimuth, elevation) from microphone
            frequency: Frequency of the sound (optional)
        
        Returns:
            Recorded pressure value
        """
        azimuth, elevation = source_direction
        
        # Apply directivity
        directivity_coeff = self.get_directivity(azimuth, elevation, frequency)
        recorded_pressure = pressure * directivity_coeff
        
        # Apply frequency response (simplified)
        magnitude, _ = self.frequency_response.get_response_at_frequency(
            frequency or 1000.0
        )
        recorded_pressure *= 10 ** (magnitude / 20.0)  # Convert dB to linear
        
        return recorded_pressure
    
    def record_velocity(self, velocity: np.ndarray, source_direction: Tuple[float, float],
                       frequency: Optional[float] = None) -> np.ndarray:
        """
        Record velocity vector with directivity applied.
        
        Args:
            velocity: Incident velocity vector (vx, vy, vz)
            source_direction: Tuple of (azimuth, elevation) from microphone
            frequency: Frequency of the sound (optional)
        
        Returns:
            Recorded velocity vector
        """
        azimuth, elevation = source_direction
        
        # Apply directivity to velocity magnitude
        directivity_coeff = self.get_directivity(azimuth, elevation, frequency)
        recorded_velocity = velocity * directivity_coeff
        
        return recorded_velocity
    
    def update_position(self, frame: int, position_data: np.ndarray):
        """Update microphone position and orientation for current frame"""
        if frame < len(position_data):
            # Position data format: [x, y, z, qx, qy, qz, qw] or Euler angles
            frame_data = position_data[frame]
            
            if len(frame_data) >= 3:
                self.position = frame_data[:3]
            
            if len(frame_data) >= 7:
                # Quaternion format
                self.orientation = frame_data[3:7]
            elif len(frame_data) >= 6:
                # Euler angles format
                self.orientation = self._euler_to_quaternion(frame_data[3:6])
    
    def _euler_to_quaternion(self, euler_angles: np.ndarray) -> np.ndarray:
        """Convert Euler angles (roll, pitch, yaw) to quaternion"""
        roll, pitch, yaw = euler_angles
        
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw])
    
    def get_global_direction(self, local_azimuth: float, local_elevation: float) -> Tuple[float, float]:
        """
        Convert local direction to global coordinates using microphone orientation.
        
        Args:
            local_azimuth: Local azimuth relative to microphone
            local_elevation: Local elevation relative to microphone
        
        Returns:
            Global (azimuth, elevation) tuple
        """
        # Simplified implementation - in practice use quaternion rotation
        # For omnidirectional, orientation doesn't affect directivity
        return local_azimuth, local_elevation
    
    def apply_calibration(self, audio_signal: np.ndarray, 
                         frequency: Optional[float] = None) -> np.ndarray:
        """
        Apply complete calibration to recorded audio signal.
        
        Args:
            audio_signal: Recorded audio signal
            frequency: Center frequency (optional)
        
        Returns:
            Calibrated audio signal
        """
        return self.calibration.apply_calibration(audio_signal, frequency)

