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

from dask import delayed, compute

from ..core.entity_manager import EntityManager

from ..lib.frequency_bands import FrequencyBands
from ..lib.acoustic_object import AcousticObject

from ..sources.spherical_source import SphericalSource
from ..sources.planar_source import PlanarSource

from ..engine.wave_propagator import WavePropagator

from ..outputs.ambisonic_output import AmbisonicOutput
from ..outputs.omnidirectional_output import OmnidirectionalOutput
from ..outputs.cardioid_output import CardioidOutput
from ..outputs.hypercardioid_output import HypercardioidOutput
from ..outputs.figure8_output import Figure8Output

@dataclass
class AcousticEngine:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')

        tasks = [self._add_source(source) for source in config.sources]
        tasks += [self._add_object(obj) for obj in config.objects]
        tasks += [self._add_output(output) for output in config.outputs]
        compute(*tasks)

        for i in range(len(config.sources)):
            for j in range(len(config.outputs)):
                combos.append([config.sources[i].idx, config.outputs[j].idx])
        tasks = [self._add_solvers(combo) for combo in combos]
        compute(*tasks)

    @delayed
    def _add_source(self, config):
        source_map = {
            'spherical': SphericalSource,
            'planar': PlanarSource
        }
        if 'SourceConfig' in str(type(config)) and config.type in source_map:
            source = source_map.get(config.type)(self.entity_manager, config.idx)
            self.entity_manager.register('sources', source)

    @delayed
    def _add_object(self, config):
        if 'ObjectConfig' in str(type(config)):
            obj = AcousticObject(self.entity_manager, config.idx)
            self.entity_manager.register('objects', obj)

    @delayed
    def _add_output(self, config):
        output_map = {
            'ambisonic': AmbisonicOutput,
            'omnidirectional': OmnidirectionalOutput,
            'cardioid': CardioidOutput,
            'hypercardioid': HypercardioidOutput,
            'figure8': Figure8Output
        }
        if 'OutputConfig' in str(type(config)) and config.type in output_map:
            output = output_map.get(config.type)(self.entity_manager, config.idx)
            self.entity_manager.register('outputs', output)

    @delayed
    def _add_solvers(self, combo):
        wave_propagator = WavePropagator(self.entity_manager, combo)
        self.entity_manager.register('wave_propagators', wave_propagator)

    def compute(self):
        wave_propagators = self.entity_manager.get('wave_propagators')
        tasks = [wave_propagators[index].compute() for index in wave_propagators.keys()]
        compute(*tasks)
