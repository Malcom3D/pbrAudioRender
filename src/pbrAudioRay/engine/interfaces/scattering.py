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
class ScatteringInterface:
    """Handle sound wave scattering at rough surfaces after reflections"""
    entity_manager: EntityManager

    def compute(self, material_properties: Any, primID_filtered: np.ndarray, normals: np.ndarray, ray_data: Any, new_origins: np.ndarray):
        # Compute scattering energies and phases
        scat_coeffs = material_properties.scattering_coeffs[primID_filtered][:, ray_data.bands_idx]
        scat_phases = material_properties.scattering_phases[primID_filtered][:, ray_data.bands_idx]

        # Generate scattering rays
        scattered_data = self._generate_scattering_rays(material_properties, new_origins, normals, scat_coeffs, scat_phases, ray_data)

        return scattered_data

    def _generate_scattering_rays(self, material_properties: Any, origins: np.ndarray, normals: np.ndarray, scat_coeffs: np.ndarray, scat_phases: np.ndarray, ray_data: Any) -> Dict[str, np.ndarray]:
        """Generate scattering rays on hemisphere."""
        config = self.entity_manager.get('config')
        n_scat_origins = origins.shape[0]
        max_scattering = int(config.interface.max_scattering*np.mean(ray_data.energies))

        if max_scattering < 1:
            return {
                'origins': np.zeros((0, 3), dtype=np.float32),
                'directions': np.zeros((0, 3), dtype=np.float32),
                'normals': np.zeros((0, 3), dtype=np.float32),
                'energies': np.zeros((0, 1), dtype=np.float32),
                'phases': np.zeros((0, 1), dtype=np.float32),
                'delay': np.zeros((0, 1), dtype=np.float32)
            }
        elif max_scattering == 1:
            n_scat_rays = np.full((n_scat_origins, 1), [1], dtype=np.int32)
        else:
            n_scat_rays = np.random.randint(1, max_scattering, size=(n_scat_origins, 1))

        # Generate number of scattering rays
        roughness = material_properties.roughness
        n_samples = np.sum(n_scat_rays)

        # Initialize arrays
        result = {
            'origins': np.zeros((n_samples, 3), dtype=np.float32),
            'directions': np.zeros((n_samples, 3), dtype=np.float32),
            'normals': np.zeros((n_samples, 3), dtype=np.float32),
            'energies': np.zeros((n_samples, 1), dtype=np.float32),
            'phases': np.zeros((n_samples, 1), dtype=np.float32),
            'delay': np.zeros((n_samples, 1), dtype=np.float32)
        }

        # Generate random directions on hemisphere
        hi_idx = 0
        for idx in range(n_scat_origins):
            lo_idx = hi_idx
            hi_idx = lo_idx + int(n_scat_rays[idx])
            n_rays_this = hi_idx - lo_idx

            # Copy info array
            result['origins'][lo_idx:hi_idx] = origins[idx]
            result['directions'][lo_idx:hi_idx] = ray_data.directions[idx]
            result['normals'][lo_idx:hi_idx] = normals[idx]
            result['energies'][lo_idx:hi_idx] = ray_data.energies[idx] * scat_coeffs[idx].reshape(-1,1)
            result['phases'][lo_idx:hi_idx] = ray_data.phases[idx] * -scat_phases[idx].reshape(-1, 1) % (2 * np.pi)
            result['delay'][lo_idx:hi_idx] = ray_data.delay[idx]

            # Generate random directions on hemisphere
            random_dirs = np.random.uniform(-1, 1, (n_rays_this, 3))
            random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)

            # Ensure directions point along hemisphere oriented by normal
            normal = normals[idx]
            dot_products = np.sum(random_dirs * normal, axis=1)
            flip_mask = dot_products < 0
            random_dirs[flip_mask] = -random_dirs[flip_mask]

            result['directions'][lo_idx:hi_idx] = random_dirs

            # Distribute energy among scattering rays
            result['energies'][lo_idx:hi_idx] = scat_coeffs[idx] / n_rays_this

        return result

