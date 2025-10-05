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
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            data = json.load(f)
            self.simulation = SimulationConfig(**data['config'])
            self.sources = SourceConfig(**data['sources'])
            self.outputs = OutputConfig(**data['outputs'])

@dataclass
class SimulationConfig:
    sample_rate: int = 48000
    voxel_size: float = 0.1  # meters
    grid_size: tuple = (100, 100, 100)  # x, y, z
    speed_of_sound: float = 343.0  # m/s
    air_density: float = 1.225  # kg/m³
    max_frames: int = 48000  # 1 second at 48kHz
    pml_thickness: int = 10
    absorption_coeff: float = 0.99
    frequency_range: tuple[float, float] = (20.0, 20000.0)
    ambisonic_order: int = 3
    
@dataclass
class SourceConfig:
    type: str  # "spherical" or "plane"
    position: tuple  # (x, y, z)
    rotation: tuple  # (roll, pitch, yaw) in radians
    audio_file: str
    gain: float = 1.0

@dataclass
class OutputConfig:
    position: tuple
    spatial_arrangement: str  # path to JSON file
    ambisonic_order: int = 3
