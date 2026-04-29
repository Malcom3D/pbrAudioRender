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

@dataclass
class ReflectionInterface:
    """Handle rays reflection at objects boundaries"""
    entity_manager: EntityManager

    def compute(self, normals: np.ndarray, directions: np.ndarray, energies: np.ndarray, phases: np.ndarray, refl_coeffs: np.ndarray, refl_phases: np.ndarray, ray_data: Any):
        # Compute incident angles and reflected directions
        dot = np.sum(directions * normals, axis=1)
        incident_angles = np.arccos(-dot)
        reflected_directions = directions - 2 * dot[:, np.newaxis] * normals

        # Compute intersection reflection energies and phase shift
        reflected_energy = energies * refl_coeffs
        reflected_phase = phases + refl_phases % (2 * np.pi)

        return reflected_energy, reflected_phase, incident_angles, reflected_directions
