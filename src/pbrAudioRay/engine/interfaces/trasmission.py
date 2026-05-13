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
class TransmissionInterface:
    entity_manager: EntityManager

    def compute(self, medium_objs: np.ndarray, material_properties: Any, primID_filtered: np.ndarray, normals: np.ndarray, ray_data: Any, absorbed_energies: np.ndarray):
        # Compute transmission energies and phases
        trans_coeffs = material_properties.transmission_coeffs[primID_filtered][:, ray_data.bands_idx]
        trans_phases = material_properties.transmission_phases[primID_filtered][:, ray_data.bands_idx]

        # Compute transmission origins
        transmission_origins = inters - (0.01 * normals)
        transmission_origins = transmission_origins.astype(np.float32)

        # Compute incident angles
        dot_projection = np.sum(ray_data.directions * normals, axis=1)
        incident_angles = np.arccos(-dot_projection)

        # Compute incident directions
        incident_directions = ray_data.origins - transmission_origins

        # Generate transmission rays
        scattered_data = self._generate_transmission_rays(transmission_origins, normals, trans_coeffs, trans_phases, incident_angles, incident_directions)

    def _generate_transmission_rays(self, origins: np.ndarray, normals: np.ndarray, trans_coeffs: np.ndarray, trans_phases: np.ndarray, incident_angles: np.ndarray, incident_directions: np.ndarray) -> Dict[str, np.ndarray]:
        # Compute normal incident transmission (no refraction)
        # Compute oblique incident transmission (Snell's Law refraction)
        pass
