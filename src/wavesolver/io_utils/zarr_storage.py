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

import zarr
import numpy as np
from typing import Tuple, Dict, Any, Optional
import os

class ZarrStoreManager:
    """Manages zarr storage for soxel grid data with metadata support."""
    
    def __init__(self, store_path: str = "simulation_data.zarr", mode: str = 'w'):
        self.store_path = store_path
        self.mode = mode
        self.store = None
        self.metadata = {}
        
    def initialize_store(self, dimensions: Tuple[int, int, int], voxel_size: float):
        """Initialize a new zarr store for simulation data."""
        self.store = zarr.open(self.store_path, mode=self.mode)
        
        # Store simulation metadata
        self.metadata = {
            'dimensions': dimensions,
            'voxel_size': voxel_size,
            'num_frames': 0,
            'simulation_parameters': {}
        }
        
        # Create arrays for pressure and velocity
        self.store.create_dataset('pressure', 
                                shape=(0, *dimensions), 
                                chunks=(1, *dimensions),
                                dtype=np.float32)
        
        self.store.create_dataset('velocity_x', 
                                shape=(0, *dimensions), 
                                chunks=(1, *dimensions),
                                dtype=np.float32)
        self.store.create_dataset('velocity_y', 
                                shape=(0, *dimensions), 
                                chunks=(1, *dimensions),
                                dtype=np.float32)
        self.store.create_dataset('velocity_z', 
                                shape=(0, *dimensions), 
                                chunks=(1, *dimensions),
                                dtype=np.float32)
        
        # Store metadata
        self.store.attrs.update(self.metadata)
    
    def open_existing_store(self, store_path: str):
        """Open an existing zarr store."""
        self.store_path = store_path
        self.store = zarr.open(store_path, mode='r')
        self.metadata = dict(self.store.attrs)
        return self.metadata
    
    def add_simulation_parameters(self, params: Dict[str, Any]):
        """Add simulation parameters to metadata."""
        if 'simulation_parameters' not in self.metadata:
            self.metadata['simulation_parameters'] = {}
        
        self.metadata['simulation_parameters'].update(params)
        if self.store:
            self.store.attrs.update(self.metadata)
    
    def add_frame(self, pressure: np.ndarray, velocity: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Add a new frame to the store."""
        if self.store is None:
            raise RuntimeError("Store not initialized. Call initialize_store first.")
        
        vx, vy, vz = velocity
        
        # Append data
        self.store['pressure'].append(pressure[np.newaxis, ...])
        self.store['velocity_x'].append(vx[np.newaxis, ...])
        self.store['velocity_y'].append(vy[np.newaxis, ...])
        self.store['velocity_z'].append(vz[np.newaxis, ...])
        
        # Update frame count
        self.metadata['num_frames'] += 1
        self.store.attrs['num_frames'] = self.metadata['num_frames']
    
    def get_frame(self, frame_idx: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Retrieve a specific frame from the store."""
        if self.store is None:
            raise RuntimeError("Store not opened.")
        
        pressure = self.store['pressure'][frame_idx]
        vx = self.store['velocity_x'][frame_idx]
        vy = self.store['velocity_y'][frame_idx]
        vz = self.store['velocity_z'][frame_idx]
        
        return pressure, (vx, vy, vz)
    
    def get_time_series_at_position(self, position: Tuple[int, int, int]) -> np.ndarray:
        """Get pressure time series at a specific grid position."""
        if self.store is None:
            raise RuntimeError("Store not opened.")
        
        i, j, k = position
        pressure_series = self.store['pressure'][:, i, j, k]
        return pressure_series
    
    def close(self):
        """Close the store (mainly for file-based stores)."""
        # Zarr stores don't typically need explicit closing for file-based stores
        # but this method is here for interface consistency
        pass
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get information about the store."""
        if self.store is None:
            return {}
        
        info = {
            'store_path': self.store_path,
            'dimensions': self.metadata.get('dimensions'),
            'voxel_size': self.metadata.get('voxel_size'),
            'num_frames': self.metadata.get('num_frames', 0),
            'data_size_gb': self._calculate_store_size()
        }
        return info
    
    def _calculate_store_size(self) -> float:
        """Calculate approximate store size in GB."""
        if self.store is None:
            return 0.0
        
        total_size = 0
        for key in ['pressure', 'velocity_x', 'velocity_y', 'velocity_z']:
                       if key in self.store:
                array = self.store[key]
                total_size += array.nbytes
        
        return total_size / (1024 ** 3)  # Convert to GB
