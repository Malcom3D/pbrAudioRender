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
Figure-8 microphone output with bidirectional directivity pattern.
"""

import numpy as np
from typing import Dict, List, Tuple Tuple, Optional, Any
from dataclasses import dataclass
from .omnidirectional_output import OmnidirectionalOutput
from ..lib.directivity_pattern import DirectivityPattern, CommonDirectivityPatterns
from ..lib.frequency_response import FrequencyResponse
from ..lib.calibration import OutputCalibration


@dataclass
class Figure8Output(OmnidirectionalOutput):
    """Figure-8 microphone with bidirectional directivity pattern"""
    
    def __init__(self, output_config):
        super().__init__(output_config)
        
        # Override with figure-8 directivity pattern
        if (not output_config.directivity_pattern and 
            not output_config.directivity_pattern_file):
            self.directivity_pattern = CommonDirectivityPatterns.figure8()
    
    def get_directivity(self, azimuth: float, elevation: float, 
                       frequency: Optional[float] = None) -> float:
        """
        Get figure-8 directivity coefficient.
        Figure-8 pattern: cos(θ)
        where θ is the angle from the microphone's front axis.
        """
        # Convert global direction to local coordinates
        local_azimuth, local_elevation = self._global_to_local_direction(azimuth, elevation)
        
        # Figure-8 pattern is primarily azimuth-dependent
        theta = np.deg2rad(local_azimuth)
        
        # Classic figure-8 pattern
        directivity = np.cos(theta)
        
        # Optional: Apply frequency-dependent variations
        if frequency is not None:
            # Figure-8 pattern typically maintains shape across frequencies
            # but may have some high-frequency lobing
            if frequency > 8000:
                # Simple model for high-frequency lobing
                lobing_factor = np.sin(2 * theta) * 0.1
                directivity += lobing_factor
        
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
        # Simplified implementation
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
        # Figure-8 has nulls at 90° and 270° from front
        return [(90.0, 0.0), (270.0, 0.0)]
    
    def get_p_polar_response(self, frequencies: List[float] = None) -> Dict[float, np.ndarray]:
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
)
            for i, az in enumerate(azimuths):
                response[i] = self.get_directivity(az, 0, freq)  # On-axis elevation
            polar_responses[freq] = response
        
        return polar_responses

