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
from typing import Dict, Any, List, Optional, Tuple
import dask.array as da

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

from .core.wave_solver import WaveSolver3D
from .core.boundary_conditions import BoundaryConditions
from .core.modal_synthesis import ModalSynthesizer
from .sources.wave_source import SphericalWaveSource, PlaneWaveSource, ModalSource
from .io_utils.zarr_store import SoxelZarrStore
from .io_utils.openvdb_exporter import OpenVDBExporter
from .io_utils.ambisonic_renderer import AmbisonicRenderer
from .utils.config import SimulationConfig, PhysicalProperties

class SoxelEngine:
    """Main acoustic engine class"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.wave_solver = WaveSolver3D(config)
        self.boundary_conditions = BoundaryConditions()
        self.modal_synthesizer = ModalSynthesizer({})
        
        # Initialize storage
        self.zarr_store = SoxelZarrStore("./soxel_data.zarr", config)
        self.vdb_exporter = OpenVDBExporter(config)
        self.ambisonic_renderer = AmbisonicRenderer(
            order=config.ambisonics_order,
            sample_rate=config.sample_rate
        )
        
        # Simulation state
        self.current_frame = 0
        self.sources = []
        self._initialize_simulation_grid()

        # Exporter data
#        self.ambisonic_array = []
#        self.vdb_frame_index = 0
    
    def _initialize_simulation_grid(self):
        """Initialize simulation grid with PML"""
        grid_shape = self.config.grid_shape
        
        # Pressure field (double buffered)
        self.pressure = np.zeros((2,) + grid_shape, dtype=np.float32)
        
        # Velocity fields
        self.velocity_x = np.zeros((2,) + grid_shape, dtype=np.float32)
        self.velocity_y = np.zeros((2,) + grid_shape, dtype=np.float32)
        self.velocity_z = np.zeros((2,) + grid_shape, dtype=np.float32)
        
        # Material properties
        self.material_map = np.zeros(grid_shape, dtype=np.int32)
        self.impedance_map = np.ones((256,), dtype=np.float32) * 415.0  # Default air impedance
        
        # Initialize with some obstacles
        self._setup_test_scene()
    
    def _setup_test_scene(self):
        """Setup a simple test scene with some obstacles"""
        grid_shape = self.config.grid_shape
        center = np.array(grid_shape) // 2
        
        # Add a sphere obstacle
        radius = 15
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    dist = np.sqrt((i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2)
                    if dist < radius:
                        self.material_map[i, j, k] = 1  # Solid material
                        self.impedance_map[1] = 1e10   # High impedance (rigid)
    
    def _add_spherical_source(self, position: Tuple[float, float, float], 
                            audio_data: Optional[np.ndarray] = None, **kwargs):
        """Add spherical wave source"""
        source = SphericalWaveSource(
            position=position,
            amplitude=kwargs.get('amplitude', 1.0),
            frequency=kwargs.get('frequency', None),
            audio_data=audio_data
        )
        self.sources.append(source)
    
    def _add_plane_source(self, position: Tuple[float, float, float],
                         audio_data: Optional[np.ndarray] = None, **kwargs):
        """Add plane wave source"""
        source = PlaneWaveSource(
            position=position,
            direction=kwargs.get('direction', (1, 0, 0)),
            amplitude=kwargs.get('amplitude', 1.0),
            frequency=kwargs.get('frequency', None),
            audio_data=audio_data
        )
        self.sources.append(source)
    
    def _add_modal_source(self, position: Tuple[float, float, float], **kwargs):
        """Add modal synthesis source"""
        material_props = kwargs.get('material_properties', {
            'youngs_modulus': 2e9,
            'poisson_ratio': 0.3,
            'density': 1000.0
        })
        
        # Create simple geometry around position
        geometry = self._create_modal_geometry(position)
        
        source = ModalSource(
            position=position,
            material_properties=material_props,
            geometry=geometry
        )
        self.sources.append(source)
    
    def _create_modal_geometry(self, position: Tuple[float, float, float]) -> np.ndarray:
        """Create simple geometry for modal analysis"""
        # Create a simple rectangular geometry around the position
        x, y, z = position
        size = 5
        points = []
        
        for i in range(-size, size+1):
            for j in range(-size, size+1):
                for k in range(-size, size+1):
                    if abs(i) + abs(j) + abs(k) <= size:  # Diamond shape
                        points.append([x + i, y + j, z + k])
        
        return np.array(points, dtype=np.float32)
    
    def add_wave_source(self, source_type: str, position: Tuple[float, float, float],
                       audio_data: Optional[np.ndarray] = None, **kwargs):
        """Add a wave source to the simulation"""
        if source_type == "spherical":
            self._add_spherical_source(position, audio_data, **kwargs)
        elif source_type == "plane":
            self._add_plane_source(position, audio_data, **kwargs)
        elif source_type == "modal":
            self._add_modal_source(position, **kwargs)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
    
    def _apply_sources(self, current_time: float):
        """Apply all active sources to the pressure field"""
        dt = self.config.dt
        
        for source in self.sources:
            if source.active:
                source_value = source.get_pressure(current_time, dt)
                
                # Convert position to grid coordinates
                grid_pos = tuple(int(p / self.config.voxel_size) for p in source.position)
                
                # Apply source to pressure field
                self.wave_solver.apply_source(
                    self.pressure, grid_pos, source_value
                )
    
    def simulate_frame(self):
        """Simulate one frame of wave propagation"""
        current_time = self.current_frame * self.config.dt
        
        self.pressure[1] = np.zeros((1,) + self.config.grid_shape, dtype=np.float32)

        # Apply sources
        self._apply_sources(current_time)
        
        # Update equations
        self.wave_solver.update_pressure_3d(
            self.pressure, self.velocity_x, self.velocity_y, self.velocity_z,
            self.config.dt, self.config.voxel_size, self.config.speed_of_sound,
            1.225, self.impedance_map, self.material_map
        )
        
        self.wave_solver.update_velocity_3d(
            self.velocity_x, self.velocity_y, self.velocity_z, self.pressure,
            self.config.dt, self.config.voxel_size, 1.225
        )
        
        # Apply boundary conditions
        self.boundary_conditions.apply_pml_3d(
            self.pressure, self.config.pml_thickness
        )
        
        # Store frame
        frame_data = {
            'pressure': self.pressure[1],  # New pressure field
            'velocity_x': self.velocity_x[1],
            'velocity_y': self.velocity_y[1],
            'velocity_z': self.velocity_z[1]
        }
        self.zarr_store.store_frame(frame_data, self.current_frame)
        
        # Export to OpenVDB periodically
        if self.current_frame % 1 == 0:  # Export every 1 frames
            filename = f"frame_{self.current_frame:06d}.vdb"
            filename = os.path.join('out_openvdb', filename)
            self.vdb_exporter.export_frame(self.pressure[1], self.current_frame, filename)
        
        # Swap buffers
        self.pressure[0] = self.pressure[1]
        self.velocity_x[0] = self.velocity_x[1]
        self.velocity_y[0] = self.velocity_y[1]
        self.velocity_z[0] = self.velocity_z[1]
        
        self.current_frame += 1
    
    def render_ambisonic_output(self, listener_positions: List[Tuple[float, float, float]],
                               output_file: str):
        """Render ambisonic output for given listener positions"""
        # Implementation would read from zarr store and render ambisonic output
        print(f"Rendering ambisonic output to {output_file}")
        # Actual implementation would process stored frames and create audio file
        #ambisonic_array = self.ambisonic_renderer.pressure_to_ambisonics(self.pressure, listener_positions)
        #np.savez_compressed(output_file, array1=ambisonic_array)
