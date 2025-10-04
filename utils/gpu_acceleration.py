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

import numba
import numpy as np
from numba import cuda, jit
import zarr
import zarrs
import dask.array as da

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

class GPUConfig:
    def __init__(self):
        self.has_cuda = cuda.is_available()
        self.has_gpu = self.has_cuda
        
    def configure_numba(self):
        if self.has_cuda:
            numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = False
            return "cuda"
        return "cpu"
    
    def configure_dask(self):
        if self.has_cuda:
            import dask_cudf
            return "cuda"
        return "cpu"
    
    def get_parallel_target(self):
        return "cuda" if self.has_cuda else "parallel"
