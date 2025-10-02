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
import openvdb as vdb
from typing import Dict, Any

from ..utils.config import SimulationConfig, SimulationConfig

class OpenVDBExporter:
    """Export simulation frames to OpenVDB for visualization"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def frame_to_vdb(self, pressure_field: np.ndarray, frame_index: int) -> vdb.FloatGrid:
        """Convert pressure field to OpenVDB grid"""
        # Create VDB grid
        grid = vdb.FloatGrid()
        grid.copyFromArray(pressure_field)
        
        # Set grid name and transform
        grid.name = f"pressure_frame_{frame_index:06d}"
        
        # Set voxel size
        transform = vdb.createLinearTransform(voxelSize=self.config.voxel_size)
        grid.transform = transform
        
        return grid
    
    def export_frame(self, pressure_field: np.ndarray, frame_index: int, 
                    filename: str):
        """Export single frame to VDB file"""
        grid = self.frame_to_vdb(pressure_field, frame_index)
        
        # Write to file
        vdb.write(filename, grids=[grid])
