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

from ..core.entity_manager import EntityManager
from ..lib.ray_tracer import RayTracer

@dataclass
class WavePropagator:
    """Main wave propagator manager coordinating all physical processes"""
    entity_manager: EntityManager
    combo: Tuple[int, int]
    source_idx: int = None
    output_idx: int = None

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.source_idx = combo[0]
        self.output_idx = combo[1]

    def compute(self):
        pass
