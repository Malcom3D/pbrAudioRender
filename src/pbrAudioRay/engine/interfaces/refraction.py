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
import numba as nb
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ...core.entity_manager import EntityManager

@staticmethod
@nb.jit(nopython=True)
def _snells_law(incident_angle: float, sound_speed1: float, sound_speed2: float) -> float:
    """Calculate refraction angle using Snell's Law"""
    # Snell's Law: sin(θ1)/c1 = sin(θ2)/c2
    sin_theta2 = (sound_speed2 / sound_speed1) * np.sin(incident__angle)

    # Handle total internal reflection
    if abs(sin_theta2) > 1.0:
        return np.pi / 2  # Total internal reflection

    return np.arcsin(sin_theta2)

@dataclass
class RefractionInterface:
    """Handle rays refraction at objects boundaries"""
    entity_manager: EntityManager

    def compute(self):
        pass
