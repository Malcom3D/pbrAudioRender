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
from typing import Dict, Any

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

class SoxelZarrStore:
    """Zarr-based storage for soxel data"""
    
    def __init__(self, store_path: str, config, 
                 compressors=zarr.codecs.BloscCodec(cname='blosclz', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)):
        self.store_path = store_path
        self.config = config
        self.compressors = compressors
        
        # Initialize zarr store
#        self.root = zarr.open(store_path, mode='w')
        self.root = zarr.create_group(store=store_path)
        
        # Create datasets
        self._initialize_datasets()
    
    def _initialize_datasets(self):
        """Initialize zarr datasets for soxel data"""
        grid_shape = self.config.grid_shape
        
        # Pressure field (time, x, y, z)
#        self.root.zeros('pressure', 
        self.pressure = self.root.create_array(
                       name='pressure', 
                       shape=(0,) + grid_shape,
                       chunks=(1,) + grid_shape,
                       dtype=np.float32,
                       compressors=self.compressors)
        
        # Velocity fields
#        for component in ['velocity_x', 'velocity_y', 'velocity_z']:
#            self.root.zeros(component,
#            self.root.create_array(
#                           name=component,
#                           shape=(0,) + grid_shape,
#                           chunks=(1,) + grid_shape,
#                           dtype=np.float32,
#                           compressors=self.compressors)

        # go manual...
        self.velocity_x = self.root.create_array(
                       name='velocity_x',
                       shape=(0,) + grid_shape,
                       chunks=(1,) + grid_shape,
                       dtype=np.float32,
                       compressors=self.compressors)

        # go manual...
        self.velocity_y = self.root.create_array(
                       name='velocity_y',
                       shape=(0,) + grid_shape,
                       chunks=(1,) + grid_shape,
                       dtype=np.float32,
                       compressors=self.compressors)

        # go manual...
        self.velocity_z = self.root.create_array(
                       name='velocity_z',
                       shape=(0,) + grid_shape,
                       chunks=(1,) + grid_shape,
                       dtype=np.float32,
                       compressors=self.compressors)
        
        # Material properties
#        self.root.zeros('material_map',
        self.material_map = self.root.create_array(
                       name='material_map',
                       shape=grid_shape,
                       chunks=(64, 64, 64),
                       dtype=np.int32,
                       compressors=self.compressors)
        
        # Metadata
        self.root.attrs['sample_rate'] = self.config.sample_rate
        self.root.attrs['voxel_size'] = self.config.voxel_size
        self.root.attrs['grid_shape'] = self.config.grid_shape
    
    def store_frame(self, frame_data: Dict[str, np.ndarray], frame_index: int):
        """Store a single frame of simulation data"""
#        print('zarr_store.store_frame: ', frame_data, frame_index)
        for field_name, data in frame_data.items():
            for name, array in self.root.arrays():
                if field_name in name:
                    dataset = array
                    # Append to existing dataset
                    if len(data.shape) == 3:  # Spatial field
                        data = data[np.newaxis, ...]  # Add time dimension
                        print('zarr_store.store_frame: ', data)
                        filename_npz = field_name + '.npz'
                        np.savez_compressed(filename_npz, array1=data)
                    dataset.append(data)
