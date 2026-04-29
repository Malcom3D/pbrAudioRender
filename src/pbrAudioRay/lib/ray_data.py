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
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class RayData:
    src_idx: int
    out_idx: int
    bands_idx: int
    recursion_idx: int
    origins: np.ndarray
    directions: np.ndarray
    energies: np.ndarray
    phases: np.ndarray
    hits_coords: np.ndarray = None
    path_length: np.ndarray = None
    medium_absorption_coeffs: np.ndarray = None
    medium_absorption_phases: np.ndarray = None
    hits: Dict[str, np.ndarray] = None
