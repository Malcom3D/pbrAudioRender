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
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.acoustic_layer import AcousticLayer
from ..lib.acoustic_field import AcousticField, FrequencyLimitedField, VelocityVectors

@dataclass
class LayerManager:
    entity_manager: EntityManager
    idx: int
    ended: bool = field(default=False)
    layers : Dict[int, AcousticLayer] = field(default_factory=dict)

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.shape = config.acoustic_domain.shape
        self.elements_map = {
            'pressure': 'pressure',
            'vx': 'velocity.vx',
            'vy': 'velocity.vy',
            'vz': 'velocity.vz'
        }

    def add_new(self, name: str, bands_idx: int) -> None:
        if self.get_layer(name, bands_idx) == None:
            new_layer = AcousticLayer(name, bands_idx, shape=self.shape)
            self.layers[len(self.layers)] = new_layer

    def len_by_name(self, name: str, bands_idx: int = None) -> int:
        layers_num = 0
        for index in self.layers.keys():
            if bands_idx == None and name in self.layers[index].name:
                layers_num += 1
            elif bands_idx == self.layers[index].bands_idx and name in self.layers[index].name:
                layers_num += 1
        return layers_num

    def get_layer(self, name: str, bands_idx: int) -> np.ndarray:
        for index in self.layers.keys():
            if name in self.layers[index].name and bands_idx == self.layers[index].bands_idx:
                return self.layers.get(index)

    def get_array(self, name: str, bands_idx: int, element: str) -> np.ndarray:
        layer = self.get_layer(name, bands_idx)

        for key in self.elements_map.keys():
            if element in key:
                val = self.elements_map.get(key)

        grid = np.empty(self.shape, dtype=float)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    grid[i,j,k] = eval(f"layer.field[i,j,k].{val}")

        return grid

    def update_layer(self, name: str, bands_idx: int, low_freq: float, high_freq: float, new_pressure: np.ndarray, new_vx: np.ndarray, new_vy: np.ndarray, new_vz: np.ndarray):
        layer = self.get_layer(name, bands_idx)
        for i in range(layer.shape[0]):
            for j in range(layer.shape[1]):
                for k in range(layer.shape[2]):
                    velocity_vectors = VelocityVectors(new_vx[i,j,k],new_vy[i,j,k],new_vz[i,j,k])
                    layer.field[i,j,k] = FrequencyLimitedField(low_freq=low_freq, high_freq=high_freq, pressure=new_pressure[i,j,k], velocity=velocity_vectors)
