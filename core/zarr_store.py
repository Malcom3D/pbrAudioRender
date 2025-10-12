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
from typing import Dict, Any, Optional
import os

class ZarrStore:
    """Manages Zarr storage for simulation data"""
    
    def __init__(self, config):
        self.config = config
        self.store_path = "./data/simulation.zarr"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        
        # Initialize Zarr store
        self.store = zarr.open(self.store_path, mode='a')
        
        # Create groups
        if 'soxel_grid' not in self.store:
            self.store.create_group('soxel_grid')
        if 'layer_managers' not in self.store:
            self.store.create_group('layer_managers')
        
        # Store configuration
        self._store_configuration()
    
    def _store_configuration(self):
        """Store simulation configuration in Zarr"""
        if 'configuration' not in self.store:
            config_group = self.store.create_group('configuration')
            
            # Store basic configuration
            config_group.array('grid_shape', data=self.config.voxel_grid.shape)
            config_group.array('voxel_size', data=[self.config.voxel_grid.voxel_size])
            config_group.array('sample_rate', data=[self.config.voxel_grid.sample_rate])
            
            # Store source information
            sources_group = config_group.create_group('sources')
            for source in self.config.sources:
                source_group = sources_group.create_group(f'source_{source.idx}')
                source_group.array('name', data=[source.name])
                source_group.array('type', data=[source.type])
    
    def save_SoxelGrid(self, soxel_grid, frame: int):
        """Save SoxelGrid state to Zarr"""
        group = self.store['soxel_grid']
        
        # Convert Soxel grid to saveable format
        grid_data = self._soxel_grid_to_array(soxel_grid)
        
        # Save as new array
        array_name = f"frame_{frame:06d}"
        if array_name not in group:
            group.create_dataset(
                array_name,
                data=grid_data,
                shape=soxel_grid.shape,
                dtype=np.float32,
                chunks=(32, 32, 32),
                compression='zlib'
            )
        else:
            group[array_name][:] = grid_data
    
    def save_LayerManager(self, layer_manager, frame: int):
        """Save LayerManager state to Zarr"""
        group = self.store['layer_managers']
        source_group_name = f"source_{layer_manager.source_idx}"
        
        if source_group_name not in group:
            group.create_group(source_group_name)
        
        source_group = group[source_group_name]
        
        # Save all fields
        fields = layer_manager.get_fields()
        for field_name, field_data in fields.items():
            array_name = f"{field_name}_{frame:06d}"
            if array_name not in source_group:
                source_group.create_dataset(
                    array_name,
                    data=field_data,
                    shape=layer_manager.shape,
                    dtype=np.float32,
                    chunks=(32, 32, 32),
                    compression='zlib'
                )
            else:
                source_group[array_name][:] = field_data
    
    def _soxel_grid_to_array(self, soxel_grid) -> np.ndarray:
        """Convert SoxelGrid to numpy array for storage"""
        # Store sound speed as representative value
        grid_data = np.zeros(soxel_grid.shape, dtype=np.float32)
        
        for i in range(soxel_grid.shape[0]):
            for j in range(soxel_grid.shape[1]):
                for k in range(soxel_grid.shape[2]):
                    soxel = soxel_grid.grid[i, j, k]
                    grid_data[i, j, k] = soxel.sound_speed
        
        return grid_data

    def get_soxel_grid_data(self, frame: int) -> Optional[np.ndarray]:
        """Retrieve SoxelGrid data for specific frame"""
        try:
            group = self.store['soxel_grid']
            array_name = f"frame_{frame:06d}"
            if array_name in group:
                               return group[array_name][:]
            else:
                return None
        except KeyError:
            return None
    
    def get_layer_manager_data(self, source_idx: int, frame: int) -> Dict[str, np.ndarray]:
        """Retrieve LayerManager data for specific source and frame"""
        try:
            group = self.store['layer_managers'][f"source_{source_idx}"]
            fields = {}
            
            for field_name in ['pressure', 'velocity_x', 'velocity_y', 'velocity_z']:
                array_name = f"{field_name}_{frame:06d}"
                if array_name in group:
                    fields[field_name] = group[array_name][:]
                else:
                    # Return zeros if data doesn't exist
                    shape = self.config.voxel_grid.shape
                    fields[field_name] = np.zeros(shape, dtype=np.float32)
            
            return fields
            
        except KeyError:
            # Return empty fields if source group doesn't exist
            shape = self.config.voxel_grid.shape
            return {
                'pressure': np.zeros(shape, dtype=np.float32),
                'velocity_x': np.zeros(shape, dtype=np.float32),
                'velocity_y': np.zeros(shape, dtype=np.float32),
                'velocity_z': np.zeros(shape, dtype=np.float32)
            }
    
    def get_available_frames(self) -> Dict[str, list]:
        """Get list of available frames for each data type"""
        available = {
            'soxel_grid': [],
            'layer_managers': {}
        }
        
        # Get SoxelGrid frames
        soxel_group = self.store['soxel_grid']
        for key in soxel_group.keys():
            if key.startswith('frame_'):
                frame_num = int(key.split('_')[1])
                available['soxel_grid'].append(frame_num)
        
        # Get LayerManager frames
        layer_group = self.store['layer_managers']
        for source_key in layer_group.keys():
            if source_key.startswith('source_'):
                source_idx = int(source_key.split('_')[1])
                available['layer_managers'][source_idx] = []
                
                source_data = layer_group[source_key]
                for field_key in source_data.keys():
                    if field_key.startswith('pressure_'):
                        frame_num = int(field_key.split('_')[1])
                        available['layer_managers'][source_idx].append(frame_num)
        
        return available
    
    def cleanup_old_frames(self, keep_last_n: int = 1000):
        """Clean up old frames to save storage space"""
        available = self.get_available_frames()
        
        # Clean SoxelGrid frames
        soxel_frames = sorted(available['soxel_grid'])
        frames_to_remove = soxel_frames[:-keep_last_n] if len(soxel_frames) > keep_last_n else []
        
        for frame in frames_to_remove:
            array_name = f"frame_{frame:06d}"
            if array_name in self.store['soxel_grid']:
                del self.store['soxel_grid'][array_name]
        
        # Clean LayerManager frames
        for source_idx, frames in available['layer_managers'].items():
            frames_to_remove = sorted(frames)[:-keep_last_n] if len(frames) > keep_last_n else []
            
            for frame in frames_to_remove:
                source_group = self.store['layer_managers'][f"source_{source_idx}"]
                for field in ['pressure', 'velocity_x', 'velocity_y', 'velocity_z']:
                    array_name = f"{field}_{frame:06d}"
                    if array_name in source_group:
                        del source_group[array_name]
        
        if frames_to_remove:
            print(f"Cleaned up {len(frames_to_remove)} old frames")

