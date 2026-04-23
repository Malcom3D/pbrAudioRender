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

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from dask import delayed, compute

from ..core.entity_manager import EntityManager

from ..outputs.omnidirectional_output import OmnidirectionalOutput
from ..outputs.cardioid_output import CardioidOutput
from ..outputs.hypercardioid_output import HypercardioidOutput
from ..outputs.figure8_output import Figure8Output

from ..lib.functions import _get_position, _get_rotation, _world_to_grid, _cartesian_to_spherical
from ..lib.interpolator import FrequencyInterpolator

@dataclass
class AmbisonicOutput:
    """Ambisonic microphones arrangement output"""
    entity_manager: EntityManager
    idx: int

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.output_config = None

        for output_config in config.outputs:
            if output_config.idx == self.idx:
                self.output_config = output_config

        # Frequency bands for processing
        frequency_bands = self.entity_manager.get('frequency_bands')
        self.bands = frequency_bands.get_bands()

        # Load spatial arrangement configuration
        with open(self.output_config.spatial_arrangement_file, 'r') as f:
            self.spatial_config = json.load(f)

        tasks = [self._add_output(output) for output in self.spatial_config['outputs']]
        compute(*tasks)

    @delayed
    def _add_output(self, config):
        output_map = {
            'omnidirectional': OmnidirectionalOutput,
            'cardioid': CardioidOutput,
            'hypercardioid': HypercardioidOutput,
            'figure8': Figure8Output
        }
        if config['type'] in output_map:
            output = output_map.get(config['type'])(self.entity_manager, self.idx, config['id'])
            idx = int(f"{(1+self.idx)*1000}{config['id']}")
            self.entity_manager.register('outputs', output, idx)

    def get_mics(self):
        mics = []
        for output in self.spatial_config['outputs']:
            idx = int(f"{(1+self.idx)*1000}{output['id']}")
            mics.append(self.entity_manager.get('outputs', idx))
        return mics

    def process_audio(self)-> np.ndarray:
        samples = []
        mics = self.get_mics()
        for mic in mics:
            sample = mic.process_audio()
            samples.append(sample)
        return samples
