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

    def compute(self, material_properties: Any, primID_filtered: np.ndarray, normals: np.ndarray, ray_data: Any, new_origins: np.ndarray):
        refl_coeffs = material_properties.reflection_coeffs[primID_filtered][:, ray_data.bands_idx]
        refl_phases = material_properties.reflection_phases[primID_filtered][:, ray_data.bands_idx]

        # Compute incident angles
        dot_projection = np.sum(ray_data.directions * normals, axis=1)
        incident_angles = np.arccos(-dot_projection)

        # Compute reflection energies and phases
        reflected_energies = ray_data.energies * refl_coeffs.reshape(-1, 1)
        reflected_phases = ray_data.phases * -refl_phases.reshape(-1, 1) % (2 * np.pi)

        # Compute incident directions
        incident_directions = ray_data.origins - new_origins
        reflected_directions = self._compute_reflection_directions(incident_directions, normals, incident_angles)

    return reflected_energies, reflected_phases, reflected_directions

    def _compute_reflection_directions(self, incident_directions: np.ndarray, normals: np.ndarray, incident_angles: np.ndarray) -> np.ndarray:
        """Compute reflection direction vectors."""
        # Normalize inputs
        incident_directions = incident_directions / np.linalg.norm(incident_directions, axis=1, keepdims=True)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        # Compute components
        n_dot_i = np.sum(normals * incident_directions, axis=1, keepdims=True)
        incident_normals = n_dot_i * normals
        incident_tangent = incident_directions - incident_normals

        # Normalize tangent
        tangent_norm = np.linalg.norm(incident_tangent, axis=1, keepdims=True)
        incident_tangent_unit = incident_tangent / (tangent_norm + 1e-10)

        # Compute reflection direction
        reflection_directions = (np.cos(incident_angles.reshape(-1, 1)) * incident_normals - np.sin(incident_angles.reshape(-1, 1)) * incident_tangent_unit)

        return reflection_directions / np.linalg.norm(reflection_directions, axis=1, keepdims=True)

