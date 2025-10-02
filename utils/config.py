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
import zarr
import zarrs
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

@dataclass
class SimulationConfig:
    """Configuration for acoustic simulation"""
    sample_rate: int = 48000
    voxel_size: float = 0.01  # meters
    grid_shape: Tuple[int, int, int] = (256, 256, 256)
    cfl_number: float = 0.3
    pml_thickness: int = 20
    max_frequency: float = 20000.0
    ambisonics_order: int = 3
    speed_of_sound: float = 343.0
    dt: float = 1 / sample_rate
    
@dataclass
class PhysicalProperties:
    """Physical properties of acoustic medium"""
    density: float = 1.225  # kg/m³
    speed_of_sound: float = 343.0  # m/s
    impedance: float = 1.225 * 343.0  # Rayls
    absorption_coeff: np.ndarray = None  # Frequency-dependent
    reflection_coeff: np.ndarray = None
    
    def __post_init__(self):
        if self.absorption_coeff is None:
            # Default frequency-dependent absorption
            freqs = np.linspace(20, 20000, 1000)
            self.absorption_coeff = 0.1 * (freqs / 1000) ** 0.5
