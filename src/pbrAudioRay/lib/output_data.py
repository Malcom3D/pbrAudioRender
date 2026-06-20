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
from dataclasses import dataclass, field

@dataclass
class OutputData:
    """Holds accumulated output data."""
    frame_idx: int = None
    bands_idx: int = None
    source_idx: int = None
    output_idx: int = None
    energies: np.ndarray = field(default_factory=lambda: np.zeros((0, 1), dtype=np.float32))
    phases: np.ndarray = field(default_factory=lambda: np.zeros((0, 1), dtype=np.float32))
    delay: np.ndarray = field(default_factory=lambda: np.zeros((0, 1), dtype=np.float32))
    origins: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    directions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
