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

from dataclasses import dataclass
from typing import List, Tuple

from core.entity_manager import EntityManager

from core.soxel_grid import SoxelGrid
from lib.frames import FrameCounter
from lib.frequency_bands import FrequencyBands
from utils.gpu_acceleration import GPUManager

from sources.spherical_source import SphericalSource
from sources.planar_source import PlanarSource

from objects.acoustic_object import AcousticObject

from engine.wave_propagator import WavePropagator

from outputs.omnidirectional_output import OmnidirectionalOutput
from outputs.cardioid_output import CardioidOutput
from outputs.hypercardioid_output import HypercardioidOutput
from outputs.figure8_output import Figure8Output

@dataclass
class AcousticEngine:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')

        self._add_singletons('gpu')
        self._add_singletons('frames')
        self._add_singletons('frequency_bands')

        for source in config.sources:
            self._add_source(source)
        for obj in config.objects:
            self._add_object(obj)
        for output in config.outputs:
            self._add_output(output)
        
        self._add_singletons('soxel_grid')

        for source in config.sources:
            self._add_solvers(source)

    def _add_source(self, config):
        source_map = {
            'spherical': SphericalSource,
            'planar': PlanarSource
        } 
        if 'SourceConfig' in str(type(config)) and config.type in source_map:
            source = source_map.get(config.type)(self.entity_manager, config.idx)
            self.entity_manager.register('sources', source, config.idx)

    def _add_object(self, config):
        if 'ObjectConfig' in str(type(config)):
            obj = AcousticObject(self.entity_manager, config.idx)
            self.entity_manager.register('objects', obj, config.idx)

    def _add_output(self, config):
        output_map = {
            'omnidirectional': OmnidirectionalOutput,
            'cardioid': CardioidOutput,
            'hypercardioid': HypercardioidOutput,
            'figure8': Figure8Output
        }
        if 'OutputConfig' in str(type(config)) and config.type in output_map:
            output = output_map.get(config.type)(self.entity_manager, config.idx)
            self.entity_manager.register('outputs', output, config.idx)

    def _add_singletons(self, name: str):
        singletons_map = {
            'gpu': GPUManager,
            'frames': FrameCounter,
            'frequency_bands': FrequencyBands,
            'soxel_grid': SoxelGrid
        }
        for s in singletons_map.keys():
            if name in s:
                single = singletons_map.get(s)(self.entity_manager)
                self.entity_manager.register(s, single)

    def _add_solvers(self, config):
        if 'SourceConfig' in str(type(config)):
            wave_propagator = WavePropagator(self.entity_manager, config.idx)
            self.entity_manager.register('wave_propagators', wave_propagator, config.idx)

    def update(self):
        soxel_grid = self.entity_manager.get('soxel_grid')
        soxel_grid.update()
        wave_propagators = self.entity_manager.get('wave_propagators')
        for index in wave_propagators.keys():
            wave_propagator = wave_propagators.get(index)
            print("wave_propagator.update", index)
            wave_propagator.update()
        frames = self.entity_manager.get('frames')
        frames.next()

