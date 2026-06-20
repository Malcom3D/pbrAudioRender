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

from pbrAudioCommon.lib.import_helper import np
import numba as nb
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ...core.entity_manager import EntityManager

@dataclass
class AbsorptionInterface:
    entity_manager: EntityManager

    def compute(self, material_properties: Any, primID_filtered: np.ndarray, normals: np.ndarray, ray_data: Any):
        abs_coeffs = material_properties.absorption_coeffs[primID_filtered][:, ray_data.bands_idx]
        abs_phases = material_properties.absorption_phases[primID_filtered][:, ray_data.bands_idx]

        # Compute incident angles
        dot_projection = np.sum(ray_data.directions * normals, axis=1)
        incident_angles = np.arccos(-dot_projection)

        # Compute absorbed energies
        angle_factor = np.cos(incident_angles)
        angle_factor[angle_factor == 0] = 1e-16
        absorbed_energies = ray_data.energies * angle_factor.reshape(-1,1) * abs_coeffs.reshape(-1,1)

        return absorbed_energies
