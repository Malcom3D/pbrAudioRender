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

@dataclass
class AbsorptionInterface:
    entity_manager: EntityManager
    idx: int
    bands_idx: int

    def __post_init__(self):
        # Get low and high frequency
        frequency_bands = self.entity_manager.get('frequency_bands')
        bands = frequency_bands.get_bands()
        self.low_freq = bands[self.bands_idx][0]
        self.high_freq = bands[self.bands_idx][1]

#    @nb.jit(nopython=True, parallel=True)
    def _apply_absorption(self, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, boundaries: Dict, absorption_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply frequency-dependent absorption to fields at boundaries"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        for boundary_idx in nb.prange(len(boundaries['impedance_discontinuities'])):
            boundary = boundaries['impedance_discontinuities'][boundary_idx]
            i, j, k = boundary['position']
            
            # Get absorption coefficients for current frequency bands
            absorption_factor = absorption_coeffs[i, j, k]
            absorption_factor = max(0.0, min(1.0, absorption_factor))
            
            new_pressure[i, j, k] *= absorption_factor
            new_vx[i, j, k] *= absorption_factor
            new_vy[i, j, k] *= absorption_factor
            new_vz[i, j, k] *= absorption_factor
        
        return new_pressure, new_vx, new_vy, new_vz
    
    def update(self, boundaries: Dict[str, Any]):
        """Apply absorption to fields at boundaries"""
        config = self.entity_manager.get('config')
        enable_absorption = config.interface.enable_absorption
        if not enable_absorption:
            return

        wave_propagator = self.entity_manager.get('wave_propagators', self.idx)
        layer_manager = wave_propagator.layer_manager
        soxel_grid = self.entity_manager.get('soxel_grid')

        absorption_coeffs = soxel_grid.get_array('absorption', self.low_freq, self.high_freq)

        names = []
        items = list(layer_manager.layers.items())
        for index, item in items:
            if not item.name in names and item.bands_idx == self.bands_idx:
                name = item.name
                names.append(name)
        
                # Apply absorption
                new_pressure, new_vx, new_vy, new_vz = self._apply_absorption(
                    layer_manager.get_array(name, self.bands_idx, 'pressure'),
                    layer_manager.get_array(name, self.bands_idx, 'vx'),
                    layer_manager.get_array(name, self.bands_idx, 'vy'),
                    layer_manager.get_array(name, self.bands_idx, 'vz'),
                    boundaries,
                    absorption_coeffs
                )

                # Apply (new_pressure, new_vx, new_vy, new_vz) to selected layer
                layer = layer_manager.get_layer(name, self.bands_idx)
                layer_manager.update_layer(layer.name, layer.bands_idx, self.low_freq, self.high_freq, new_pressure, new_vx, new_vy, new_vz)
