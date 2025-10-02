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
import zarr
from typing import Tuple, Optional
import numba

class SoxelGrid:
    """
    Represents a colocated acoustic 3D grid data structure using zarr for storage.
    """
    
    def __init__(self, dimensions: Tuple[int, int, int], 
                 voxel_size: float, 
                 zarr_store_path: str = "simulation_data.zarr"):
        self.dimensions = dimensions
        self.voxel_size = voxel_size
        self.zarr_store_path = zarr_store_path
        
        # Initialize zarr store
        self.store = zarr.open(zarr_store_path, mode='w')
        
        # Create datasets for pressure and velocity fields
        self.pressure = self.store.zeros('pressure', 
                                       shape=(0, *dimensions), 
                                       chunks=(1, *dimensions),
                                       dtype=np.float32)
        
        self.velocity_x = self.store.zeros('velocity_x', 
                                         shape=(0, *dimensions), 
                                         chunks=(1, *dimensions),
                                         dtype=np.float32)
        self.velocity_y = self.store.zeros('velocity_y', 
                                         shape=(0, *dimensions), 
                                         chunks=(1, *dimensions),
                                         dtype=np.float32)
        self.velocity_z = self.store.zeros('velocity_z', 
                                         shape=(0, *dimensions), 
                                         chunks=(1, *dimensions),
                                         dtype=np.float32)
        
        self.current_frame = 0
    
    def add_frame(self, pressure: np.ndarray, 
                  velocity: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Add a new frame to the zarrarr store."""
        if pressure.shape != self.dimensions:
            raise ValueError(f"Pressure shape {pressure.shape} doesn't match grid dimensions {self.dimensions}")
        
        # Append pressure
        self.pressure.append(pressure[np.newaxis, ...])
        
        # Append velocity components
        vx, vy, vz = velocity
        self.velocity_x.append(vx[np.newaxis, ...])
        self.velocity_y.append(vy[np.newaxis, ...])
        self.velocity_z.append(vz[np.newaxis, ...])
        
        self.current_frame += 1
    
    def get_frame(self, frame_idx: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Retrieve a specific frame from the zarr store."""
        pressure = self.pressure[frame_idx]
        vx = self.velocity_x[frame_idx]
        vy = self.velocity_y[frame_idx]
        vz = self.velocity_z[frame_idx]
        
        return pressure, (vx, vy, vz)
    
    @property
    def num_frames(self) -> int:
        return self.current_frame
