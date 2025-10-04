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
from typing import Optional
import zarr
import zarrs
import openvdb as vdb

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

class OpenVDBExporter:
    def __init__(self):
        try:
#            import pyopenvdb as vdb
            self.vdb = vdb
            self.has_openvdb = True
        except ImportError:
            self.has_openvdb = False
            print("Warning: OpenVDB not available. Install with 'pip install openvdb'")
    
    def export_frame(self, soxels, frame: int, filename: str):
        """Export soxel data to OpenVDB format"""
        if not self.has_openvdb:
            return
            
        # Extract pressure and velocity fields
        pressure_grid = self.create_scalar_grid(soxels, 'pressure', frame)
        velocity_grid = self.create_vector_grid(soxels, 'velocity', frame)
        
        # Create VDB file
        self.vdb.write(filename, grids=[pressure_grid, velocity_grid])
        
    def create_scalar_grid(self, soxels, field: str, frame: int):
        """Create scalar grid for pressure"""
        grid_shape = soxels.shape
        data = np.zeros(grid_shape, dtype=np.float32)
        
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    if field == 'pressure':
                        data[i,j,k] = soxels[i,j,k].pressure
        
        grid = self.vdb.FloatGrid()
        grid.copyFromArray(data)
        grid.name = f"{field}"
        
        return grid
    
    def create_vector_grid(self, soxels, field: str, frame: int):
        """Create vector grid for velocity"""
        grid_shape = soxels.shape
        data = np.zeros(grid_shape + (3,), dtype=np.float32)
        
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    if field == 'velocity':
                        data[i,j,k] = soxels[i,j,k].velocity
        
        grid = self.vdb.Vec3SGrid()
        grid.copyFromArray(data)
        grid.name = f"{field}"
        
        return grid
