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
import numba as nb
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ...core.entity_manager import EntityManager
from ...lib.ray_data import RayData

@dataclass
class AbsorptionInterface:
    entity_manager: EntityManager
    
    def compute(self, ray: List[RayData]):
        """Apply frequency-dependent absorption."""
        # get fraquency bands
        frequency_bands = frequency_bands.get_bands()

        # Get ray data
        obj_idx = ray.object_idx
        bands_idx = ray.bands_idx
        low_freq, high_freq = frequency_bands[bands_idx]
        shader = ray.medium_shader

        # Get object config
        objs_config = self.entity_manager.get('objects')
        for c_idx in objs_config.keys():
            if objs_config[c_idx].idx == obj_idx:
                obj_config = objs_config[c_idx]

        # Absorption: reduce energy
        if shader.acoustic_properties and shader.acoustic_properties.absorption:
            # Get absorption coefficient at given frequency 
            coeffs = shader.acoustic_properties.absorption.get_avg_coeffs(low_freq, high_freq)
            ray.energy *= (1 - coeffs)

        return ray
