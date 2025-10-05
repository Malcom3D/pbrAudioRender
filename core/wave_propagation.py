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
import numpy as np
import zarr
import zarrs
from numba import jit
from typing import List, Dict
from .soxel import Soxel, SoxelType, PhysicalProperties
from ..engine.fdtd_solver import FDTDSolver
from ..engine.boundary_conditions import BoundaryConditions
from ..engine.zarr_store import ZarrStoreManager

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

class WavePropagation:
    def __init__(self, config, gpu_config):
        self.config = config
        self.gpu_config = gpu_config
        self.soxels = None
        self.fdtd_solver = None
        self.boundary_conditions = None
        self.zarr_store = None
        
        self.setup_engine()
        
    def setup_engine(self):
        """Initialize all engine components"""
        grid_shape = self.config.grid_size
        
        # Initialize soxel grid
        self.soxels = np.empty(grid_shape, dtype=object)
        self.initialize_soxel_grid()
        
        # Initialize solvers
        self.fdtd_solver = FDTDSolver(
            grid_shape, 
            self.config.voxel_size,
            self.config.sample_rate,
            self.config.speed_of_sound,
            self.gpu_config
        )
        
        self.boundary_conditions = BoundaryConditions(
            grid_shape,
            self.config.pml_thickness,
            self.gpu_config
        )
        
        self.zarr_store = ZarrStoreManager(
            grid_shape,
            self.config.max_frames,
            f"simulation_{self.config.sample_rate}hz"
        )
        
    def initialize_soxel_grid(self):
        """Initialize the soxel grid with default air properties"""
        for i in range(self.config.grid_size[0]):
            for j in range(self.config.grid_size[1]):
                for k in range(self.config.grid_size[2]):
                    air_props = PhysicalProperties(
                        speed_of_sound=self.config.speed_of_sound,
                        density=self.config.air_density,
                        absorption_coeff=np.zeros(100),  # frequency bins
                        reflection_coeff=np.zeros(100),
                        impedance=np.full(100, 413.0)  # air impedance
                    )
                    self.soxels[i,j,k] = Soxel(SoxelType.MEDIUM, (i,j,k), air_props)
    
    def add_source(self, source_config, source_obj):
        """Add a sound source to the grid"""
        x, y, z = source_config.position
        self.soxels[x,y,z].type = SoxelType.SOURCE
        self.soxels[x,y,z].source_obj = source_obj
        
    def add_output(self, output_config, output_obj):
        """Add an output point to the grid"""
        x, y, z = output_config.position
        self.soxels[x,y,z].type = SoxelType.OUTPUT
        self.soxels[x,y,z].output_obj = output_obj
        
    def run_simulation(self, sources: List, outputs: List):
        """Main simulation loop"""
        current_frame = 0
        
        while current_frame < self.config.max_frames:
            # Update sources
            self.update_sources(sources, current_frame)
            
            # Solve wave propagation
            pressure_field, velocity_field = self.fdtd_solver.solve_time_step(
                self.soxels, current_frame
            )
            
            # Apply boundary conditions
            pressure_field, velocity_field = self.boundary_conditions.apply(
                pressure_field, velocity_field
            )
            
            # Update soxels
            self.update_soxel_fields(pressure_field, velocity_field)
            
            # Store frame data
            self.zarr_store.store_frame(
                current_frame, pressure_field, velocity_field
            )
            
            # Export visualization frame
            if current_frame % 1 == 0:  # Export every 1 frames
                self.export_visualization_frame(current_frame)
                
            current_frame += 1
            
    def update_sources(self, sources: List, frame: int):
        """Update source pressures for current frame"""
        for source in sources:
            x, y, z = source.position
            sample = source.get_sample(frame)
            self.soxels[x,y,z].pressure += sample
            
    def update_soxel_fields(self, pressure_field: np.ndarray, velocity_field: np.ndarray):
        """Update soxel pressures and velocities from field arrays"""
        for i in range(self.config.grid_size[0]):
            for j in range(self.config.grid_size[1]):
                for k in range(self.config.grid_size[2]):
                    self.soxels[i,j,k].pressure = pressure_field[i,j,k]
                    self.soxels[i,j,k].velocity = velocity_field[i,j,k,:]
                    
    def export_visualization_frame(self, frame: int):
        """Export frame for visualization"""
        from ..renderer.openvdb_exporter import OpenVDBExporter
        exporter = OpenVDBExporter()
        if not os.path.exists('data/vdb'):
            os.makedirs('data/vdb')
        exporter.export_frame(
            self.soxels, 
            frame, 
            f"data/vdb/frame_{frame:06d}.vdb"
        )
