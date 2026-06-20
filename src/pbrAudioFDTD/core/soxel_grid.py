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

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import math
import os

from ..core.entity_manager import EntityManager
from ..lib.soxel import Soxel

from ..lib.acoustic_shader import AcousticShader
from ..lib.acoustic_field import AcousticField

from ..sources.spherical_source import SphericalSource
from ..sources.planar_source import PlanarSource

class SoxelGrid():
    """Manages the 3D grid of soxels and their acoustic properties"""
    def __init__(self, entity_manager: EntityManager, **kwargs):

        self.entity_manager = entity_manager
        self.config = entity_manager.get('config')
        self.frames = entity_manager.get('frames')

        self.idx = self.config.acoustic_domain.idx
        self.shape = self.config.acoustic_domain.shape
        self.default_shader = self.config.acoustic_domain.acoustic_shader

        # Initialize the grid
#        self._initialize_grid()

    def _initialize_grid(self):
        """Initialize soxel grid"""
        self.soxels = np.empty(self.shape, dtype=object)

        """Initialize the soxel grid with default medium"""
        # Fill grid with default medium
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    self.soxels[i, j, k] = Soxel(
                        idx=self.idx,
                        type = 0,                  # mark as default_medium
                        input_pressures = None,
                        acoustic_shader = self.default_shader
                    )

    def _init_sources(self):
        # Write soxels from sources to the grid
        for source_config in self.config.sources:
            source = self.entity_manager.get('sources', source_config.idx)
            soxel_list = source.get_soxels()
            for i,j,k, soxel in soxel_list:
                self.soxels[i,j,k] = soxel

    def _init_objects(self):
        # Write soxels from acoustic objects to the grid
        for obj_config in self.config.objects:
            obj = self.entity_manager.get('objects', obj_config.idx)
            soxel_list = obj.get_soxels()
            for i,j,k, soxel in soxel_list:
                self.soxels[i,j,k] = soxel

    def get_array(self, element: str, low_freq: float = None, high_freq: float = None) -> np.ndarray:
        elements_map = {
            'type': 'type',
            'sound_speed': 'acoustic_shader.sound_speed',
            'density': 'acoustic_shader.density',
            'pressure': 'pressure',
            'vx': 'velocity.vx',
            'vy': 'velocity.vy',
            'vz': 'velocity.vz',
            'absorption': 'acoustic_shader.acoustic_properties.absorption',
            'refraction': 'acoustic_shader.acoustic_properties.refraction',
            'reflection': 'acoustic_shader.acoustic_properties.reflection',
            'scattering': 'acoustic_shader.acoustic_properties.scattering'
        }

        for key in elements_map.keys():
            if element in key:
                val = elements_map.get(key)

        grid = np.empty(self.shape, dtype=float)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    if self.soxels[i,j,k].type == 1 and ('pressure' in val or 'vx' in val or 'vy' in val or 'vz' in val):
                        if not low_freq == None and not high_freq == None:
                            for band_num in range(len(self.soxels[i,j,k].input_pressures.field)):
                                low = self.soxels[i,j,k].input_pressures.field[band_num].low_freq
                                high = self.soxels[i,j,k].input_pressures.field[band_num].high_freq
                                if low == low_freq and high == high_freq:
                                    grid[i,j,k] = eval(f"self.soxels[{i, j, k}].input_pressures.field[{band_num}].{val}")
                    elif 'pressure' in val or 'vx' in val or 'vy' in val or 'vz' in val:
                        grid[i,j,k] = 0.0
                    elif 'absorption' in val or 'refraction' in val or 'reflection' in val or 'scattering' in val:
                        if not low_freq == None and not high_freq == None:
                            if not eval(f"self.soxels[{i, j, k}].{val}") == None:
                                grid[i,j,k] = eval(f"self.soxels[{i, j, k}].{val}.get_avg_coeffs({low_freq}, {high_freq})")
                            else:
                                grid[i,j,k] = 0.0
                    else:
                        grid[i,j,k] = eval(f"self.soxels[{i, j, k}].{val}")
        return grid

    def get_input(self, source_idx: int):
        fields = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    if self.soxels[i,j,k].type == 1:
                        if self.soxels[i,j,k].idx == source_idx:
                            fields.append([[i,j,k],self.soxels[i,j,k].input_pressures])
        return fields

    def update(self):
        # Initialize the grid
        self._initialize_grid()
        self._init_sources()
        self._init_objects()
