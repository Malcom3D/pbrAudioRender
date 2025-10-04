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

class SolidShader:
    def __init__(self, solid_type: str = "concrete"):
        self.solid_type = solid_type
        self.properties = self.get_solid_properties()
    
    def get_solid_properties(self) -> PhysicalProperties:
        """Get physical properties for the specified solid"""
        if self.solid_type == "concrete":
            return PhysicalProperties(
                speed_of_sound=3100.0,
                density=2300.0,
                absorption_coeff=self.get_concrete_absorption(),
                reflection_coeff=self.get_concrete_reflection(),
                impedance=np.full(100, 7.13e6)
            )
        elif self.solid_type == "wood":
            return PhysicalProperties(
                speed_of_sound=3300.0,
                density=500.0,
                absorption_coeff=np.zeros(100),
                reflection_coeff=np.ones(100) * 0.8,
                impedance=np.full(100, 1.65e6)
            )
        else:
            # Default to concrete
            return PhysicalProperties(
                speed_of_sound=3100.0,
                density=2300.0,
                absorption_coeff=np.zeros(100),
                reflection_coeff=np.ones(100) * 0.9,
                impedance=np.full(100, 7.13e6)
            )
    
    def get_concrete_absorption(self) -> np.ndarray:
        """Calculate frequency-dependent concrete absorption"""
        freqs = np.linspace(20, 20000, 100)
        # Concrete absorption increases with frequency
        absorption = 0.1 * (freqs / 1000) ** 0.7
        return absorption
    
    def get_concrete_reflection(self) -> np.ndarray:
        """Calculate frequency-dependent concrete reflection"""
        freqs = np.linspace(20, 20000, 100)
        # Concrete reflection decreases slightly with frequency
        reflection = 0.95 - 0.1 * (freqs / 20000)
        return np.clip(reflection, 0, 1)
