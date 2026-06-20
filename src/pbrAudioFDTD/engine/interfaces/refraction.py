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

from pbrAudioCommon import np
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
    """Handle sound wave refraction at material boundaries"""
    entity_manager: EntityManager
    idx: int
    bands_idx: int

    def __post_init__(self):
        # Get low and high frequency
        frequency_bands = self.entity_manager.get('frequency_bands')
        bands = frequency_bands.get_bands()
        self.low_freq = bands[self.bands_idx][0]
        self.high_freq = bands[self.bands_idx][1]

    @staticmethod
#    @nb.jit(nopython=True, parallel=True)
    def _apply_refraction(pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                         sound_speed: np.ndarray, density: np.ndarray, boundaries: Dict,
                         min_impedance_ratio: float, max_impedance_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply refraction at sound speed discontinuities"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()

        # Process sound speed discontinuities for refraction
        for boundary_idx in nb.prange(len(boundaries.get('sound_speed_discontinuities', []))):
            boundary = boundaries['sound_speed_discontinuities'][boundary_idx]
            i, j, k = boundary['position']
            ni, nj, nk = boundary['neighbor_position']

            # Only process if we have significant pressure
            if np.abs(pressure[i, j, k]) > 1e-6:
                # Get sound speeds
                c1 = sound_speed[i, j, k]
                c2 = sound_speed[ni, nj, nk]

                # Calculate impedance ratio for transmission coefficient
                z1 = density[i, j, k] * c1
                z2 = density[ni, nj, nk] * c2
                impedance_ratio = max(z1, z2) / min(z1, z2)

                # Only process significant impedance changes
                if (impedance_ratio > min_impedance_ratio and
                    impedance_ratio < max_impedance_ratio):

                    # Estimate incident angle from velocity direction
                    velocity = np.array([vx[i, j, k], vy[i, j, k], vz[i, j, k]])
                    velocity_magnitude = np.sqrt(np.sum(velocity**2))

                    if velocity_magnitude > 1e-6:
                        # Calculate normal vector pointing from current to neighbor
                        normal = np.array([ni - i, nj - j, nk - k])
                        normal = normal / np.linalg.norm(normal)

                        velocity_dir = velocity / velocity_magnitude
                        incident_angle = np.arccos(np.abs(np.dot(velocity_dir, normal)))

                        # Calculate refraction angle using Snell's law
                        refraction_angle = _snells_law(incident_angle, c1, c2)

                        # Calculate transmission coefficient (simplified)
                        # For normal incidence: T = 2 * Z2 / (Z1 + Z2)
                        transmission_coeff = 2 * z2 / (z1 + z2)

                        # Apply refraction by modifying velocity direction
                        # This is a simplified model - in practice, you'd use proper vector rotation
                        refracted_velocity = velocity * transmission_coeff

                        # Update velocity in neighbor voxel (transmitted wave)
                        new_vx[ni, nj, nk] += refracted_velocity[0]
                        new_vy[ni, nj, nk] += refracted_velocity[1]
                        new_vz[ni, nj, nk] += refracted_velocity[2]

                        # Update pressure in neighbor voxel
                        new_pressure[ni, nj, nk] += pressure[i, j, k] * transmission_coeff

        return new_pressure, new_vx, new_vy, new_vz

    def update(self, boundaries: Dict[str, Any]):
        """Apply refraction to fields at sound speed discontinuities"""
        config = self.entity_manager.get('config')
        min_impedance_ratio = config.interface.min_impedance_ratio
        max_impedance_ratio = config.interface.max_impedance_ratio
        enable_refraction = config.interface.enable_refraction
        if not enable_refraction:
            return

        soxel_grid = self.entity_manager.get('soxel_grid')
        sound_speed = soxel_grid.get_array('sound_speed')
        density = soxel_grid.get_array('density')
        wave_propagator = self.entity_manager.get('wave_propagators', self.idx)
        layer_manager = wave_propagator.layer_manager

        names = []
        items = list(layer_manager.layers.items())
        for index, item in items:
            if not item.name in names and item.bands_idx == self.bands_idx:
                name = item.name
                names.append(name)

                # Apply refraction
                new_pressure, new_vx, new_vy, new_vz = self._apply_refraction(
                    layer_manager.get_array(name, self.bands_idx, 'pressure'),
                    layer_manager.get_array(name, self.bands_idx, 'vx'),
                    layer_manager.get_array(name, self.bands_idx, 'vy'),
                    layer_manager.get_array(name, self.bands_idx, 'vz'),
                    sound_speed,
                    density,
                    boundaries,
                    min_impedance_ratio,
                    max_impedance_ratio
                )

                # Apply (new_pressure, new_vx, new_vy, new_vz) to selected layer
                layer = layer_manager.get_layer(name, self.bands_idx)
                layer_manager.update_layer(layer.name, layer.bands_idx, self.low_freq, self.high_freq, new_pressure, new_vx, new_vy, new_vz)

