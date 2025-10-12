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
Hypercardioid microphone output with tighter directivity than cardioid.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .omnidirectional_output import OmnidirectionalOutput
from ..lib.directivity_pattern import DirectivityPattern, CommonDirectivityPatterns
from ..lib.frequency_response import FrequencyResponse
from ..lib.calibration import OutputCalibration


@dataclass
class HypercardioidOutput(OmnidirectionalOutput):
    """Hypercardioid microphone with tighter directivity pattern"""
    
    def __init__(self, output_config):
        super().__init__(output_config)
        
        # Override with hypercardioid directivity pattern
        if (not output_config.directivity_pattern and 
            not output_config.directivity_pattern_file):
            self.directivity_pattern = CommonDirectivityPatterns.hypercardioid()
    
    def get_directivity(self, azimuth: float, elevation: float, 
                       frequency: Optional[float] = None) -> float:
        """
        Get hypercardioid directivity coefficient.
        Hypercardioid pattern: 0.25 * (1 + 3*cos(θ))
        where θ is the angle from the microphone's front axis.
        """
        # Convert global direction to local coordinates
        local_azimuth, local_elevation = self._global_to_local_direction(azimuth, elevation)
        
        # Hypercardioid pattern is primarily azimuth-dependent
        theta = np.deg2rad(local_azimuth)
        
        # Classic hypercardioid pattern
        directivity = 0.25 * (1 + 3 * np.cos(theta))
        
        # Optional: Apply frequency-dependent variations
        if frequency is not None:
            # Hypercardioid becomes more directional at high frequencies
            freq_factor = min(1.5, frequency / 3000.0)  # More directional above 3kHz
            directivity = directivity ** (1.0 / (1.0 + 0.5 * freq_factor))
        
        return directivity
    
    def _global_to_local_direction(self, global_azimuth: float, 
                                 global_elevation: float) -> Tuple[float, float]:
        """
        Convert global direction to local microphone coordinates.
        
        Args:
            global_azimuth: Global azimuth angle
            global_elevation: Global elevation angle
        
        Returns:
            Local (azimuth, elevation) relative to microphone orientation
        """
        # Simplified implementation implementation
        # Assume microphone is oriented along global X-axis
        local_azimuth = (global_azimuth - 0) % 360
        local_elevation = global_elevation
        
        return local_azimuth, local_elevation
    
    def get_null_directions(self) -> List[Tuple[float, float]]:
        """
        Get directions where microphone has minimum sensitivity (nulls).
        
        Returns:
            List of (azimuth, elevation) tuples for null directions
        """
        # Hypercardioid has nulls at approximately 110° and 250° from front
        return [(110.0, 0.0), (250.0, 0.0)]
    
    def get_directivity_index(self, frequency: float = 1000.0) -> float:
        """
        Calculate directivity index (DI) in dB.
        
        Args:
            frequency: Frequency for DI calculation
        
        Returns:
            Directivity index in dB
        """
        # For hypercardioid, theoretical DI is about 6.0 dB
        # This can vary with frequency
        base_di = 6.0
        
        # Frequency-dependent adjustment
        if frequency > 5000:
            base_di += 1.0  # Slightly more directional at high frequencies
        elif frequency < 500:
            base_di -= 1.0  # Slightly less directional at low frequencies
        
        return base_di
    
    def get_polar_response(self, frequencies: List[float] = None) -> Dict[float, np.ndarray]:
        """
        Get polar response pattern for visualization.
        
        Args:
            frequencies: List of frequencies to evaluate
        
        Returns:
            Dictionary of polar responses by frequency
        """
        if frequencies is None:
            frequencies = [1000.0]  # Default to 1kHz
        
        polar_responses = {}
        azimuths = np.linspace(0, 360, 73)  # 5-degree resolution
        
        for freq in frequencies:
            response = np.zeros_like(azimuths)
            for i, az in enumerate(azimuths):
                response[i] = self.get_directivity(az, 0, freq)  # On-axis elevation
            polar_responses[freq] = response
        
        return polar_responses

