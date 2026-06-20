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

from pbrAudioCommon.lib.import_helper import np
from typing import Optional
from dataclasses import dataclass

@dataclass
class GeometryData:
    """Holds geometry and scene information."""
    vertices: np.ndarray
    faces: np.ndarray
    mesh_info: np.ndarray
    scene_info: np.ndarray
    normals: Optional[np.ndarray] = None
