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
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

from core.entity_manager import EntityManager
from lib.acoustic_layer import AcousticLayer
from lib.acoustic_field import AcousticField, FrequencyLimitedField, VelocityVectors

@dataclass
class LayerManager:
    entity_manager: EntityManager
    idx: int
    ended: bool = False
    layers : Dict[int, AcousticLayer] = field(default_factory=dict)

    def __post_init__(self):
        self.elements_map = {
            'pressure': 'pressure',
            'vx': 'velocity.x',
            'vy': 'velocity.y',
            'vz': 'velocity.z'
        }

    def add_new(self, name: str, bands_idx: int) -> None:
        config = self.entity_manager.get('config')
        self.shape = config.acoustic_domain.shape
        if self.get_layer(name, bands_idx) == None:
            new_layer = AcousticLayer(name, bands_idx, shape=self.shape)
            self.layers[len(self.layers)] = new_layer

    def len_by_name(self, name: str) -> int:
        layers_num = 0
        for index in self.layers.keys():
            if name in self.layers[index].name:
                layers_num += 1
        return layers_num

    def get_layer(self, name: str, bands_idx: int) -> np.ndarray:
        for index in self.layers.keys():
            if name in self.layers[index].name and bands_idx == self.layers[index].bands_idx:
                return self.layers.get(index)

    def get_shm_layer(self, name: str, bands_idx: int, element: str) -> mp.Array:
        layer = self.layers[layer_idx]

        for key in self.elements_map.keys():
            if element in key:
                val = self.elements_map.get(key)

        shm_flat = _np_to_shm_array(self.shape)
        shm_grid = _flat_to_3d(shm_flat, self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    shm_grid[i,j,k] = layer[i,j,k].val

        return shm_grid

'''
    def del_layer(self, layer_idx: int):
        pass

    def update_layer(self, layer_idx: int, element: str, low_freq: float, high_freq: float, shm_array: np.ndarray) -> None:
        layer = self.layers[layer_idx]
        np_array = _shm_array_to_np(shm_array)

        for key in self.elements_map.keys():
            if element in key:
                val = self.elements_map.get(key)

        print(np_array.shape)
'''
