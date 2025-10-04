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

class FluidShader:
    def __init__(self, fluid_type: str = "water"):
        self.fluid_type = fluid_type
        self.properties = self.get_fluid_properties()
    
    def get_fluid_properties(self) -> PhysicalProperties:
        """Get physical properties for the specified fluid"""
        if self.fluid_type == "water":
            return PhysicalProperties(
                speed_of_sound=1480.0,
                density=1000.0,
                absorption_coeff=self.get_water_absorption(),
                reflection_coeff=np.zeros(100),
                impedance=np.full(100, 1.48e6)
            )
        else:
            # Default to water
            return PhysicalProperties(
                speed_of_sound=1480.0,
                density=1000.0,
                absorption_coeff=np.zeros(100),
                reflection_coeff=np.zeros(100),
                impedance=np.full(100, 1.48e6)
            )
    
    def get_water_absorption(self) -> np.ndarray:
        """Calculate frequency-dependent water absorption"""
        freqs = np.linspace(20, 20000, 100)
        # Simplified water absorption model
        absorption = 0.01 * (freqs / 1000) ** 1.5
        return absorption
