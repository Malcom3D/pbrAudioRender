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
    def apply_scattering(pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, boundaries: Dict, scattering_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply scattering to fields at boundaries"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        for boundary_idx in nb.prange(len(boundaries['impedance_discontinuities'])):
            boundary = boundaries['impedance_discontinuities'][boundary_idx]
            i, j, k = boundary['position']
            
            scatter_coeff = scattering_coeffs[i, j, k]

            # Simple scattering model - redistribute energy randomly
            random_phase = np.exp(1j * 2 * np.pi * np.random.random())
            
            if scatter_coeff > 0:
                # Apply diffuse scattering
                new_pressure[i, j, k] *= (1 - scatter_coeff) + scatter_coeff * random_phase.real
                
                # Scatter velocity components
                scatter_dir = np.array([
                    np.random.random() - 0.5,
                    np.random.random() - 0.5, 
                    np.random.random() - 0.5
                ])
                scatter_dir = scatter_dir / np.linalg.norm(scatter_dir)
                
                velocity_mag = np.sqrt(vx[i,j,k]**2 + vy[i,j,k]**2 + vz[i,j,k]**2)
                scattered_velocity = scatter_dir * velocity_mag * scatter_coeff
                
                new_vx[i, j, k] = vx[i, j, k] * (1 - scatter_coeff) + scattered_velocity[0]
                new_vy[i, j, k] = vy[i, j, k] * (1 - scatter_coeff) + scattered_velocity[1]
                new_vz[i, j, k] = vz[i, j, k] * (1 - scatter_coeff) + scattered_velocity[2]
        
        return new_pressure, new_vx, new_vy, new_vz

    def update(self, boundaries: Dict[str, Any]):
        """Apply scattering to fields"""
        config = self.entity_manager.get('config')
        enable_scattering = config.interface.enable_absorption
        if not enable_scattering:
            return layer_manager
        
        wave_propagator = self.entity_manager.get('wave_propagators', self.idx)
        layer_manager = wave_propagator.layer_manager
        soxel_grid = self.entity_manager.get('soxel_grid')

        # Get scattering coefficients for current frequency
        scattering_coeffs = soxel_grid.get_array('scattering', self.low_freq, self.high_freq)
        
        names = []
        items = list(layer_manager.layers.items())
        for index, item in items:
            if 'reflection' in item.name and not item.name in names and item.bands_idx == self.bands_idx:
                name = item.name
                names.append(name)

                # Apply scattering
                new_pressure, new_vx, new_vy, new_vz = self.apply_scattering(
                    layer_manager.get_array(name, self.bands_idx, 'pressure'),
                    layer_manager.get_array(name, self.bands_idx, 'vx'),
                    layer_manager.get_array(name, self.bands_idx, 'vy'),
                    layer_manager.get_array(name, self.bands_idx, 'vz'),
                    boundaries,
                    scattering_coeffs
                )

                # Apply (new_pressure, new_vx, new_vy, new_vz) to selected layer
                layer = layer_manager.get_layer(name, self.bands_idx)
                layer_manager.update_layer(layer.name, layer.bands_idx, self.low_freq, self.high_freq, new_pressure, new_vx, new_vy, new_vz)
