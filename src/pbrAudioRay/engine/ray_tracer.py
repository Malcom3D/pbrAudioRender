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
from typing import Tuple, Optional, List, Any
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.embree_scene import EmbreeScene

@dataclass
class RayTracer:
    scene: Any # embreex.rtcore_scene.EmbreeScene

    def __post_init__(self):
        config = entity_manager.get('config')

    def compute(self, source_pos: np.ndarray, directions: np.ndarray):
        return self.scene.run(source_pos, directions, output=1)
