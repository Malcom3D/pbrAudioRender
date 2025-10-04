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
from dataclasses import dataclass
from typing import Union, Optional
from enum import Enum

class SoxelType(Enum):
    SOURCE = "source"
    OUTPUT = "output"
    MEDIUM = "medium"
    BOUNDARY = "boundary"

@dataclass
class PhysicalProperties:
    speed_of_sound: float
    density: float
    absorption_coeff: np.ndarray  # frequency-dependent
    reflection_coeff: np.ndarray  # frequency-dependent
    impedance: np.ndarray  # frequency-dependent
    
    def __post_init__(self):
        # Ensure arrays are properly shaped
        if isinstance(self.absorption_coeff, list):
            self.absorption_coeff = np.array(self.absorption_coeff)
        if isinstance(self.reflection_coeff, list):
            self.reflection_coeff = np.array(self.reflection_coeff)
        if isinstance(self.impedance, list):
            self.impedance = np.array(self.impedance)

class Soxel:
    def __init__(self, soxel_type: SoxelType, position: tuple, 
                 physical_props: Optional[PhysicalProperties] = None):
        self.type = soxel_type
        self.position = position
        self.physical_props = physical_props
        self.pressure = 0.0
        self.velocity = np.zeros(3)  # 3D velocity vector
        self.pressure_history = []
        
    def update_pressure(self, pressure: float):
        self.pressure = pressure
        self.pressure_history.append(pressure)
        
    def update_velocity(self, velocity: np.ndarray):
        self.velocity = velocity
        
    def get_impedance_at_freq(self, frequency: float) -> float:
        if self.physical_props is None:
            return 413.0  # Default air impedance
        # Interpolate impedance at given frequency
        freqs = np.linspace(20, 20000, len(self.physical_props.impedance))
        return np.interp(frequency, freqs, self.physical_props.impedance)
