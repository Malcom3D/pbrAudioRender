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
import zarr
import zarrs
import numpy as np
from numba import jit

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

class ZarrStoreManager:
    def __init__(self, grid_shape, max_frames, store_name):
        compressors = zarr.codecs.BloscCodec(cname='blosclz', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
        self.grid_shape = grid_shape
        self.max_frames = max_frames
        
        # Initialize Zarr store with compression
        self.store = os.path.join('data', store_name)
        self.root = zarr.create_group(store=self.store, overwrite=True)
        
        # Create arrays for pressure and velocity
        self.pressure_array = self.root.create_array(
            name='pressure',
            shape=(max_frames, grid_shape[0], grid_shape[1], grid_shape[2]),
            chunks=(100, 32, 32, 32),
            dtype=np.float32,
            compressors=compressors
        )
        
        self.velocity_array = self.root.create_array(
            name='velocity',
            shape=(max_frames, grid_shape[0], grid_shape[1], grid_shape[2], 3),
            chunks=(100, 32, 32, 32, 3),
            dtype=np.float32,
            compressors=compressors
        )
        
        self.current_frame = 0
        
    def store_frame(self, frame_idx: int, pressure: np.ndarray, velocity: np.ndarray):
        """Store pressure and velocity fields for a frame"""
        if frame_idx < self.max_frames:
            self.pressure_array[frame_idx] = pressure.astype(np.float32)
            self.velocity_array[frame_idx] = velocity.astype(np.float32)
            
    def get_frame(self, frame_idx: int):
        """Retrieve stored frame data"""
        if frame_idx < self.current_frame:
            return (
                self.pressure_array[frame_idx],
                self.velocity_array[frame_idx]
            )
        return None, None
