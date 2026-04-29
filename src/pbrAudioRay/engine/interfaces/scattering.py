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
class ScatteringInterface:
    """Handle sound wave scattering at rough surfaces after reflections"""
    entity_manager: EntityManager

    def compute(self, energies: np.ndarray, phases: np.ndarray, normals: np.ndarray, roughness_factor: np.ndarray, scat_coeffs: np.ndarray, scat_phases: np.ndarray):
        # Compute scattered directions (random direction in hemisphere)
        scattered_directions = self._random_hemisphere_directions(normals)

        # Compute intersection scattering energies and phase shift
        scattered_energy = energies * scat_coeffs * roughness_factor / max(scattered_directions.shape[0], 1e-10)
        scattered_phase = phases + scat_phases % (2 * np.pi)

 
    @staticmethod
#    @nb.njit(fastmath=True)
    def _random_hemisphere_directions(normals: np.ndarray) -> np.ndarray:
        """
        Generate random directions on hemispheres oriented along the normals.

        Args:
            normals: Surface normals
            n_samples: Number of samples to generate

        Returns:
            Array of sampled directions
        """
        n_samples = max(normals.shape[0], 1)
        directions = np.random.uniform(-1,1,(n_samples,3))
        directions /= np.linalg.norm(directions)

        while not np.all(directions[:, 0]**2 + directions[:, 1]**2 + directions[:, 2]**2 < 1):
            directions = np.random.uniform(-1,1,(n_samples,3))

            # Project onto hemisphere oriented along normals
            directions /= np.linalg.norm(directions)

        # Flip if pointing away from normals
        if not np.any(np.sum(directions * normals, axis=1) < 0):
            mask = np.sum(directions * normals, axis=1) < 0
            directions[mask] = -directions[mask]

        return directions
