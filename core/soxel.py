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

import numba as nb
import zarr
import zarrs
from numba import float32, int32

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

@nb.experimental.jitclass([
    ('pressure', float32[:]),
    ('velocity_x', float32[:]),
    ('velocity_y', float32[:]),
    ('velocity_z', float32[:]),
    ('material_id', int32),
    ('impedance', float32),
    ('absorption', float32[:]),
])
class Soxel:
    """Sound voxel containing acoustic data"""
    def __init__(self, buffer_size: int):
        self.pressure = np.zeros(buffer_size, dtype=np.float32)
        self.velocity_x = np.zeros(buffer_size, dtype=np.float32)
        self.velocity_y = np.zeros(buffer_size, dtype=np.float32)
        self.velocity_z = np.zeros(buffer_size, dtype=np.float32)
        self.material_id = 0
        self.impedance = 1.0
        self.absorption = np.zeros(1000, dtype=np.float32)  # Frequency bins
