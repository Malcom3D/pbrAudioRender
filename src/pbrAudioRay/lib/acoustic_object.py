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

import os
import numpy as np
import trimesh
from typing import Tuple, Optional, List, Any
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.functions import _load_mesh

@dataclass
class AcousticObject:
    """ Handle mesh object geometry and modal model analysis """
    entity_manager: EntityManager
    config_obj: Any
    obj_idx: int = None

    def __post_init__(self):
        self.obj_idx = self.config_obj.idx

    def get_mesh(self, frame_idx: int = None) -> trimesh.Trimesh:
        """ Return trimesh object """
        frame_idx = frame_idx if not frame_idx == None else 0
        vertices, normals, faces = self.get_data(frame_idx)
        return trimesh.Trimesh(vertices=vertices, vertex_normals=normals, faces=faces)

    def get_data(self, frame_idx: int = None) -> Tuple(np.ndarray, np.ndarray, np.ndarray)
        """ Return object geometry data"""
        frame_idx = frame_idx if not frame_idx == None else 0
        return _load_mesh(self.config_obj, frame_idx)
