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

from core.entity_manager import EntityManager

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
