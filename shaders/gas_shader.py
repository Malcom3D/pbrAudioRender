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
from core.soxel import PhysicalProperties

class GasShader:
    def __init__(self, gas_type: str = "air"):
        self.gas_type = gas_type
        self.properties = self.get_gas_properties()
    
    def get_gas_properties(self) -> PhysicalProperties:
        """Get physical properties for the specified gas"""
        if self.gas_type == "air":
            return PhysicalProperties(
                speed_of_sound=343.0,
                density=1.225,
                absorption_coeff=self.get_air_absorption(),
                reflection_coeff=np.zeros(100),
                impedance=np.full(100, 413.0)
            )
        elif self.gas_type == "helium":
            return PhysicalProperties(
                speed_of_sound=965.0,
                density=0.1786,
                absorption_coeff=np.zeros(100),
                reflection_coeff=np.zeros(100), 
                impedance=np.full(100, 172.0)
            )
        else:
            # Default to air
            return PhysicalProperties(
                speed_of_sound=343.0,
                density=1.225,
                absorption_coeff=np.zeros(100),
                reflection_coeff=np.zeros(100),
                impedance=np.full(100, 413.0)
            )
    
    def get_air_absorption(self) -> np.ndarray:
        """Calculate frequency-dependent air absorption"""
        freqs = np.linspace(20, 20000, 100)
        # Simplified air absorption model (dB/m)
        absorption = 0.5 * (freqs / 1000) ** 2
        return absorption
