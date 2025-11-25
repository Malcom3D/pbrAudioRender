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
from ..lib.functions import _get_position, _get_rotation, _world_to_grid, _cartesian_to_spherical
from ..lib.interpolator import FrequencyInterpolator

@dataclass
class CardioidOutput:
    """Cardioid microphone output with frequency-dependent processing"""
    entity_manager: EntityManager
    idx: int
    id: int = None # microphone id for ambisonic configuration

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.output_config = None

        for output_config in config.outputs:
            if output_config.idx == self.idx:
                self.output_config = output_config

                if not self.id == None:
                    with open(self.output_config.spatial_arrangement_file, 'r') as f:
                        spatial_config = json.load(f)
                        for mic_config in spatial_config.get('outputs', []):
                            if mic_config['id'] == self.id:
                                self.relative_pos = np.array(mic_config['position'])
                                self.relative_rot = np.array(mic_config['rotation'])

   
    def process_audio(self) -> float:
        """
        Process audio for cardioid microphone across all frequency bands and layers

        Returns:
            Float: the sum of the higher accuracy encoded pressure and velocity vector of all layers in the sub voxel interpolated recording position with sub-voxel accuracy at current frame after apply frequency response and calibration.
        """
        return 1

    def _get_position(self) -> Tuple[float, float, float]:
        """Get positions where this output should record"""
        frames = self.entity_manager.get('frames')
        current_frame = frames.get()
        position = _get_position(self.output_config.position_file, current_frame)
        if hasattr(self, 'relative_pos'):
            position = position + self.relative_pos
        return position

    def _get_orientation(self, frame: int) -> Tuple[float, float, float]:
        """Get microphone orientation for current frame"""
        frames = self.entity_manager.get('frames')
        current_frame = frames.get()
        rotation = _get_rotation(self.output_config.rotation_file, current_frame)
        if hasattr(self, 'relative_rot'):
            rotation = rotation + self.relative_pos
        return rotation

    def _get_interpolated_pressure(self, recording_position: Tuple[float, float, float]) -> float:
        """
        Get interpolated pressure from all layers with sub-voxel accuracy
        """
        pass

    def _get_interpolated_velocity(self, recording_position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Get interpolated velocity vectors from all layers with sub-voxel accuracy
        """
        pass

    def _get_directivity(self, azimuth: float, elevation: float) -> float:
        """Cardioid directivity pattern"""
        return 0.5 * (1 + np.cos(np.deg2rad(azimuth)))
