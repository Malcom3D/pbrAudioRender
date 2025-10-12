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
from typing import Dict, List
import time
import os
from .zarr_store import ZarrStore
from engine.wave_propagation import WavePropagation
from renderer.audio_recorder import AudioRecorder
from renderer.openvdb_exporter import OpenVDBExporter
from renderer.ambisonic_encoder import AmbisonicEncoder

class LayerManager:
    """Manages simulation layers for different sources"""
    
    def __init__(self, grid_shape: tuple, source_idx: int):
        self.source_idx = source_idx
        self.shape = grid_shape
        self.ended = False
        
        # Initialize pressure and velocity fields
        self.pressure = np.zeros(grid_shape, dtype=np.float32)
        self.velocity_x = np.zeros(grid_shape, dtype=np.float32)
        self.velocity_y = np.zeros(grid_shape, dtype=np.float32)
        self.velocity_z = np.zeros(grid_shape, dtype=np.float32)
    
    def get_fields(self) -> Dict[str, np.ndarray]:
        """Get all acoustic fields"""
        return {
            'pressure': self.pressure,
            'velocity_x': self.velocity_x,
            'velocity_y': self.velocity_y,
            'velocity_z': self.velocity_z
        }
    
    def set_fields(self, fields: Dict[str, np.ndarray]):
        """Set all acoustic fields"""
        self.pressure = fields.get('pressure', self.pressure).copy()
        self.velocity_x = fields.get('velocity_x', self.velocity_x).copy()
        self.velocity_y = fields.get('velocity_y', self.velocity_y).copy()
        self.velocity_z = fields.get('velocity_z', self.velocity_z).copy()
    
    def reset(self):
        """Reset all fields to zero"""
        self.pressure.fill(0)
        self.velocity_x.fill(0)
        self.velocity_y.fill(0)
        self.velocity_z.fill(0)
        self.ended = False

class AcousticEngine:
    """Main acoustic simulation engine"""
    
    def __init__(self, config, soxel_grid):
        self.config = config
        self.soxel_grid = soxel_grid
        self.current_frame = 0
        self.is_running = False
        
        # Create output directories
        self._create_directories()
        
        # Initialize components
        self.layer_managers = self._initialize_layer_managers()
        self.wave_propagators = self._initialize_wave_propagators()
        self.zarr_store = ZarrStore(config)
        self.audio_recorder = AudioRecorder(config.outputs, config.voxel_grid)
        
        if config.system.export_vdb:
            self.vdb_exporter = OpenVDBExporter(config.voxel_grid)
        
        self.ambisonic_encoder = AmbisonicEncoder(
            config.voxel_grid.sample_rate,
            config.voxel_grid.bit_depth,
            config.outputs
        )
    
    def _create_directories(self):
        """Create necessary output directories"""
        os.makedirs("./data/", exist_ok=True)
        os.makedirs("./exports/audio/", exist_ok=True)
        os.makedirs("./exports/ambisonic/", exist_ok=True)
        if self.config.system.export_vdb:
            os.makedirs(self.config.system.vdb_export_path, exist_ok=True)
    
    def _initialize_layer_managers(self) -> Dict[int, LayerManager]:
        """Initialize layer managers for all sources"""
        managers = {}
        for source in self.config.sources:
            managers[source.idx] = LayerManager(
                self.soxel_grid.shape, 
                source.idx
            )
        return managers
    
    def _initialize_wave_propagators(self) -> Dict[int, WavePropagation]:
        """Initialize wave propagators for all sources"""
        propagators = {}
        for source in self.config.sources:
            layer_manager = self.layer_managers[source.idx]
            propagators[source.idx] = WavePropagation(
                layer_manager,
                self.config
            )
        return propagators
    
    def run_simulation(self):
        """Run the main simulation loop"""
        self.is_running = True
        start_time = time.time()
        
        frame_limit = self.config.voxel_grid.sample_frame_limit
        if frame_limit is None:
            frame_limit = self._get_max_audio_length()
        
        print(f"Running simulation for {frame_limit} frames...")
        print(f"Grid size: {self.soxel_grid.shape}")
        print(f"Voxel size: {self.soxel_grid.voxel_size}m")
        print(f"Sample rate: {self.config.voxel_grid.sample_rate}Hz")
        
        for frame in range(frame_limit):
            self.current_frame = frame
            
            if frame % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame / elapsed if elapsed > 0 else 0
                print(f"Frame {frame}/{frame_limit} ({fps:.1f} fps)")
            
            # Update Soxel grid with current frame data
            self.soxel_grid.update(frame)
            
            # Save Soxel grid state to Zarr
            self.zarr_store.save_SoxelGrid(self.soxel_grid, frame)
            
            # Process wave propagation for all active sources
            active_sources = []
            for source in self.config.sources:
                source_idx = source.idx
                if not self.layer_managers[source_idx].ended:
                    active_sources.append(source)
            
            if not active_sources:
                print("All sources ended, stopping simulation")
                break
            
            for source in active_sources:
                source_idx = source.idx
                wave_propagator = self.wave_propagators[source_idx]
                layer_manager = self.layer_managers[source_idx]
                
                # Update wave propagation
                wave_propagator.update(self.soxel_grid, frame)
                
                # Save layer manager state
                if not layer_manager.ended:
                    self.zarr_store.save_LayerManager(layer_manager, frame)
            
            # Export VDB if enabled
            if hasattr(self, 'vdb_exporter'):
                self.vdb_exporter.save_frame(self.zarr_store, frame)
            
            # Record audio outputs
            self.audio_recorder.record(frame, self.zarr_store)
        
        # Finalize simulation
        self._finalize_simulation(start_time)
    
    def _get_max_audio_length(self) -> int:
        """Get maximum audio length from all sources"""
        max_length = 0
        for source in self.config.sources:
            if hasattr(self.soxel_grid, 'source_audio'):
                audio_data = self.soxel_grid.source_audio.get(source.idx)
                if audio_data is not None:
                    max_length = max(max_length, len(audio_data))
        
        return max_length if max_length > 0 else self.config.voxel_grid.sample_rate  # Default 1 second
    
    def _finalize_simulation(self, start_time: float):
        """Finalize simulation and save results"""
        end_time = time.time()
        simulation_time = end_time - start_time
        
        print(f"\nSimulation completed in {simulation_time:.2f} seconds")
        print(f"Processed {self.current_frame + 1} frames")
        print(f"Average speed: {(self.current_frame + 1) / simulation_time:.1f} fps")
        
        # Save audio recordings
        print("Saving audio recordings...")
        self.audio_recorder.save_recordings()
        
        # Encode to Ambisonic
        print("Encoding to Ambisonic B-format...")
        self.ambisonic_encoder.encode()
        
        # Save simulation statistics
        self._save_simulation_stats(simulation_time)
        
        self.is_running = False
    
    def _save_simulation_stats(self, simulation_time: float):
        """Save simulation statistics"""
        stats = {
            'total_frames': self.current_frame + 1,
            'simulation_time_seconds': simulation_time,
            'frames_per_second': (self.current_frame + 1) / simulation_time,
            'grid_shape': self.soxel_grid.shape,
            'voxel_size': self.soxel_grid.voxel_size,
            'sample_rate': self.config.voxel_grid.sample_rate,
            'sources_count': len(self.config.sources),
            'outputs_count': len(self.config.outputs),
            'objects_count': len(self.config.objects),
            'completion_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add audio recording statistics
        audio_stats = self.audio_recorder.get_recording_stats()
        stats['audio_recordings'] = audio_stats
        
        import json
        with open('./exports/simulation_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Simulation statistics saved to ./exports/simulation_stats.json")
    
    def get_progress(self) -> float:
        """Get simulation progress as percentage"""
        frame_limit = self.config.voxel_grid.sample_frame_limit or self._get_max_audio_length()
        if frame_limit == 0:
            return 0.0
        return (self.current_frame / frame_limit) * 100.0
    
    def stop_simulation(self):
        """Stop the simulation gracefully"""
        self.is_running = False
        print("Simulationulation stopped by user")

