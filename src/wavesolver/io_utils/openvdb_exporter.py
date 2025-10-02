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
import pyopenvdb as vdb
from typing import Tuple
import os

class OpenVDBExporter:
    """Exporter for OpenVDBDB files for visualization in Blender."""
    
    @staticmethod
    def pressure_to_vdb(pressure_field: np.ndarray, 
                       voxel_size: float,
                       bbox: Tuple[Tuple[float, float, float], 
                                  Tuple[float, float, float]] = None) -> vdb.FloatGrid:
        """Convert pressure field to OpenVDB grid."""
        
        # Create VDB grid
        grid = vdb.FloatGrid()
        grid.copyFromArray(pressure_field)
        
        # Set grid name and transform
        grid.name = "pressure"
        grid.transform = vdb.createLinearTransform(voxel_size=voxel_size)
        
        # Set value type
        grid.gridClass = vdb.GridClass.FOG
        
        return grid
    
    @staticmethod
    def velocity_to_vdb(velocity_field: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       voxel_size: float) -> vdb.Vec3SGrid:
        """Convert velocity field to OpenVDB vector grid."""
        
        vx, vy, vz = velocity_field
        
        # Create vector grid
        grid = vdb.Vec3SGrid()
        
        # Combine velocity components
        vector_data = np.stack([vx, vy, vz], axis=-1)
        grid.copyFromArray(vector_data)
        
        # Set grid properties
        grid.name = "velocity"
        grid.transform = vdb.createLinearTransform(voxel_size=voxel_size)
        grid.gridClass = vdb.GridClass.STAGGERED
        
        return grid
    
    @classmethod
    def export_frame(cls, soxel_grid, frame_idx: int, 
                    output_path: str, export_velocity: bool = False):
        """Export a specific frame to OpenVDB file."""
        
        # Get frame data
        pressure, velocity = soxel_grid.get_frame(frame_idx)
        
        # Create VDB file
        grids = []
        
        # Add pressure grid
        pressure_grid = cls.pressure_to_vdb(pressure, soxel_grid.voxel_size)
        grids.append(pressure_grid)
        
        # Add velocity grid if requested
        if export_velocity:
            velocity_grid = cls.velocity_to_vdb(velocity, soxel_grid.voxel_size)
            grids.append(velocity_grid)
        
        # Write to file
        vdb.write(output_path, grids)
        
        print(f"Exported frame {frame_idx} to {output_path}")
    
    @classmethod
    def export_animation(cls, soxel_grid, output_dir: str, 
                        start_frame: int = 0, end_frame: int = None,
                        export_velocity: bool = False):
        """Export animation sequence as OpenVDB files."""
        
        if end_frame is None:
            end_frame = soxel_grid.num_frames - 1
        
        os.makedirs(output_dir, exist_ok=True)
        
        for frame_idx in range(start_frame, end_frame + 1):
            output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.vdb")
            cls.export_frame(soxel_grid, frame_idx, output_path, export_velocity)
