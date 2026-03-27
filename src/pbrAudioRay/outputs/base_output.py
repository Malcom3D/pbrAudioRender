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

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.interpolator import FrequencyInterpolator

@dataclass
class BaseOutput:
    """Base class for all microphone outputs with common functionality"""
    entity_manager: EntityManager

    def _get_recording_position(self) -> Tuple[float, float, float]:
        """Get sub-voxel accurate recording position"""
        pass

    def _get_incident_angle(self) -> Tuple[float, float]:
        """ Calculate incident angle of sound wave relative to microphone orientation """
        pass

    def _get_orientation(self) -> Tuple[float, float, float]:
        """Get microphone orientation for current frame"""
        pass

    def _get_microphone_direction(self) -> np.ndarray:
        """ 
        Get the primary direction vector of the microphone
        For most microphones, this is the forward (z) direction
        """
        pass

    def _get_directivity(self, azimuth: float, elevation: float) -> float:
        """Base directivity pattern - should be overridden by subclasses"""
        return 1.0  # Omnidirectional by default
