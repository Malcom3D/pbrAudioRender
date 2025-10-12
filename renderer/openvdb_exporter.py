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

"""
OpenVDB exporter for 3D acoustic simulation data.
Exports pressure and velocity fields to OpenVDB format for visualization and post-processing.
"""

import numpy as np
import zarr
import openvdb as vdb
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime


class OpenVDBExporter:
    """Exports acoustic simulation data to OpenVDB format"""
    
    def __init__(self, voxel_config):
        self.config = voxel_config
        self.export_path = voxel_config.vdb_export_path
        self.voxel_size = voxel_config.voxel_size
        self.grid_shape = voxel_config.shape
        
        # Create export directory
        os.makedirs(self.export_path, exist_ok=True)
        
        # OpenVDB grid configuration
        self.grid_name_prefix = "acoustic_simulation"
        self.compression = vdb.Compression.ZIP  # Use ZIP compression for smaller files
        
        print(f"OpenVVDB exporter initialized: {self.export_path}")
    
    def save_frame(self, zarr_store, frame: int):
        """
        Export current simulation frame to OpenVDB format.
        
        Args:
            zarr_store: Zarr store containing simulation data
            frame: Current frame number
        """
        try:
            # Create OpenVDB file for this frame
            vdb_file = vdb.FloatGrid()
            
            # Export SoxelGrid data (material properties)
            self._export_soxel_grid(zarr_store, frame, vdb_file)
            
            # Export LayerManager data (acoustic fields)
            self._export_acoustic_fields(zarr_store, frame, vdb_file)
            
            # Save to file
            filename = f"{self.grid_name_prefix}_frame_{frame:06d}.vdb"
            filepath = os.path.join(self.export_path, filename)
            
            vdb.write(filepath, [vdb_file])
            
            if frame % 100 == 0:
                print(f"Exported OpenVDB frame {frame}: {filepath}")
                
        except Exception as e:
            print(f"Error exporting OpenVDB frame {frame}: {e}")
    
    def _export_soxel_grid(self, zarr_store, frame: int, vdb_file: vdb.FloatGrid):
        """Export SoxelGrid material properties to OpenVDB"""
        try:
            # Get SoxelGrid data from Zarr store
            soxel_data = zarr_store.get_soxel_grid_data(frame)
            if soxel_data is None:
                return
            
            # Create grids for different material properties
            self._create_material_grids(soxel_data, frame, vdb_file)
            
        except Exception as e:
            print(f"Error exporting SoxelGrid data for frame {frame}: {e}")
    
    def _create_material_grids(self, soxel_data: np.ndarray, frame: int, vdb_file: vdb.FloatGrid):
        """Create OpenVDB grids for material properties"""
        
        # Sound speed grid
        sound_speed_grid = vdb.FloatGrid()
        sound_speed_grid.name = f"sound_speed_{frame:06d}"
        sound_speed_grid.copyFromArray(soxel_data)
        sound_speed_grid.transform = self._create_transform()
        
        # Add to file
        vdb_file.copyFrom(sound_speed_grid)
        
        # You can add more material property grids here:
        # - Density
        # - Absorption coefficients
        # - Reflection coefficients
        # - Scattering coefficients
    
    def _export_acoustic_fields(self, zarr_store, frame: int, vdb_file: vdb.FloatGrid):
        """Export acoustic pressure and velocity fields to OpenVDB"""
        try:
            # Get all active sources
            active_sources = self._get_active_sources(zarr_store, frame)
            
            # Export pressure fields
            pressure_grid = self._create_pressure_grid(zarr_store, frame, active_sources)
            if pressure_grid is not None:
                vdb_file.copyFrom(pressure_grid)
            
            # Export velocity fields
            velocity_grids = self._create_velocity_grids(zarr_store, frame, active_sources)
            for grid in velocity_grids:
                vdb_file.copyFrom(grid)
            
            # Export combined energy field
            energy_grid = self._create_energy_grid(zarr_store, frame, active_sources)
            if energy_grid is not None:
                vdb_file.copyFrom(energy_grid)
                
        except Exception as e:
            print(f"Error exporting acoustic fields for frame {frame}: {e}")
    
    def _get_active_sources(self, zarr_store, frame: int) -> List[int]:
        """Get list of active sources for current frame"""
        active_sources = []
        
        # Check each source's LayerManager data
        for source_config in self.config.sources:
            source_idx = source_config.idx
            layer_data = zarr_store.get_layer_manager_data(source_idx, frame)
            
            if layer_data and self._has_meaningful_data(layer_data):
                active_sources.append(source_idx)
        
        return active_sources
    
    def _has_meaningful_data(self, layer_data: Dict[str, np.ndarray]) -> bool:
        """Check if layer data contains meaningful (non-zero) acoustic energy"""
        pressure = layer_data.get('pressure', np.array([]))
        if len(pressure) == 0:
            return False
        
        # Check if there's significant energy
        energy = np.sum(pressure ** 2)
        return energy > 1e-10
    
    def _create_pressure_grid(self, zarr_store, frame: int, active_sources: List[int]) -> Optional[vdb.FloatGrid]:
        """Create OpenVDB grid for combined pressure field"""
        if not active_sources:
            return None
        
        # Combine pressure from all active sources
        combined_pressure = np.zeros(self.grid_shape, dtype=np.float32)
        
        for source_idx in active_sources:
            layer_data = zarr_store.get_layer_manager_data(source_idx, frame)
            if layer_data and 'pressure' in layer_data:
                combined_pressure += layer_data['pressure']
        
        # Create OpenVDB grid
        pressure_grid = vdb.FloatGrid()
        pressure_grid.name = f"pressure_{frame:06d}"
        pressure_grid.copyFromArray(combined_pressure)
        pressure_grid.transform = self._create_transform()
        
        # Set grid metadata
        pressure_grid.insertMeta("description", "Acoustic pressure field")
        pressure_grid.insertMeta("units", "Pa")
        pressure_grid.insertMeta("frame", frame)
        pressure_grid.insertMeta("export_time", datetime.now().isoformat())
        
        return pressure_grid
    
    def _create_velocity_grids(self, zarr_store, frame: int, active_sources: List[int]) -> List[vdb.FloatGrid]:
        """Create OpenVDB grids for velocity components"""
        velocity_grids = []
        
        if not active_sources:
            return velocity_grids
        
        # Combine velocity from all active sources
        combined_vx = np.zeros(self.grid_shape, dtype=np.float32)
        combined_vy = np.zeros(self.grid_shape, dtype=np.float32)
        combined_vz = np.zeros(self.grid_shape, dtype=np.float32)
        
        for source_idx in active_sources:
            layer_data = zarr_store.get_layer_manager_data(source_idx, frame)
            if layer_data:
                combined_vx += layer_data.get('velocity_x', np.zeros(self.grid_shape))
                combined_vy += layer_data.get('velocity_y', np.zeros(self.grid_shape))
                combined_vz += layer_data.get('velocity_z', np.zeros(self.grid_shape))
        
        # Create velocity component grids
        components = [
            ('velocity_x', combined_vx),
            ('velocity_y', combined_vy),
            ('velocity_z', combined_vz)
        ]
        
        for comp_name, comp_data in components:
            grid = vdb.FloatGrid()
            grid.name = f"{comp_name}_{frame:06d}"
            grid.copyFromArray(comp_data)
            grid.transform = self._create_transform()
            
            grid.insertMeta("description", f"Acoustic {comp_name} component")
            grid.insertMeta("units", "m/s")
            grid.insertMeta("frame", frame)
            
            velocity_grids.append(grid)
        
        return velocity_grids
    
    def _create_energy_grid(self, zarr_store, frame: int, active_sources: List[int]) -> Optional[vdb.FloatGrid]:
        """Create OpenVDB grid for acoustic energy density"""
        if not active_sources:
            return None
        
        # Calculate acoustic energy density: E = 0.5 * (p²/ρc² + ρv²)
        energy_density = np.zeros(self.grid_shape, dtype=np.float32)
        
        for source_idx in active_sources:
            layer_data = zarr_store.get_layer_manager_data(source_idx, frame)
            if not layer_data:
                continue
            
            pressure = layer_data.get('pressure', np.zeros(self.grid_shape))
            vx = layer_data.get('velocity_x', np.zeros(self.grid_shape))
            vy = layer_data.get('velocity_y', np.zeros(self.grid_shape))
            vz = layer_data.get('velocity_z', np.zeros(self.grid_shape))
            
            # Use default properties for energy calculation

            sound_speed = 343.0  # m/s
            density = 1.2       # kg/m³
            
            pressure_energy = pressure ** 2 / (density * sound_speed ** 2)
            velocity_energy = density * (vx ** 2 + vy ** 2 + vz ** 2)
            
            source_energy = 0.5 * (pressure_energy + velocity_energy)
            energy_density += source_energy
        
        # Create energy grid
        energy_grid = vdb.FloatGrid()
        energy_grid.name = f"energy_density_{frame:06d}"
        energy_grid.copyFromArray(energy_density)
        energy_grid.transform = self._create_transform()
        
        energy_grid.insertMeta("description", "Acoustic energy density")
        energy_grid.insertMeta("units", "J/m³")
        energy_grid.insertMeta("frame", frame)
        
        return energy_grid
    
    def _create_transform(self) -> vdb.Transform:
        """Create OpenVDB transform with correct voxel size and orientation"""
        transform = vdb.Transform()
        
        # Set voxel size
        transform.preScale([self.voxel_size] * 3)
        
        # Set grid origin to center
        center_offset = np.array(self.grid_shape) * self.voxel_size / 2
        transform.postTranslate([-center_offset[0], -center_offset[1], -center_offset[2]])
        
        return transform
    
    def create_animation_metadata(self, total_frames: int):
        """Create metadata file for animation sequence"""
        metadata = {
            "animation_name": self.grid_name_prefix,
            "total_frames": total_frames,
            "voxel_size": self.voxel_size,
            "grid_shape": self.grid_shape,
            "export_path": self.export_path,
            "created": datetime.now().isoformat(),
            "fields": [
                {
                    "name": "pressure",
                    "description": "Acoustic pressure field",
                    "units": "Pa",
                    "data_type": "float32"
                },
                {
                    "name": "velocity_x",
                    "description": "X-component of acoustic velocity",
                    "units": "m/s",
                    "data_type": "float32"
                },
                {
                    "name": "velocity_y",
                    "description": "Y-component of acoustic velocity",
                    "units": "m/s",
                    "data_type": "float32"
                },
                {
                    "name": "velocity_z",
                    "description": "Z-component of acoustic velocity",
                    "units": "m/s",
                    "data_type": "float32"
                },
                {
                    "name": "energy_density",
                    "description": "Acoustic energy density",
                    "units": "J/m³",
                    "data_type": "float32"
                },
                {
                    "name": "sound_speed",
                    "description": "Speed of sound in medium",
                    "units": "m/s",
                    "data_type": "float32"
                }
            ]
        }
        
        metadata_file = os.path.join(self.export_path, "animation_metadata.json")
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Animation metadata saved: {metadata_file}")
    
    def export_static_scene(self, zarr_store, frame: int = 0):
        """
        Export a comprehensive static scene with all simulation data.
        
        Args:
            zarr_store: Zarr store containing simulation data
            frame: Frame to export (default: 0)
        """
        try:
            # Create a comprehensive VDB file with multiple grids
            vdb_grids = []
            
            # Export material properties
            soxel_data = zarr_store.get_soxel_grid_data(frame)
            if soxel_data is not None:
                material_grid = vdb.FloatGrid()
                material_grid.name = "material_properties"
                material_grid.copyFromArray(soxel_data)
                material_grid.transform = self._create_transform()
                vdb_grids.append(material_grid)
            
            # Export acoustic fields
            active_sources = self._get_active_sources(zarr_store, frame)
            
            pressure_grid = self._create_pressure_grid(zarr_store, frame, active_sources)
            if pressure_grid:
                vdb_grids.append(pressure_grid)
            
            velocity_grids = self._create_velocity_grids(zarr_store, frame, active_sources)
            vdb_grids.extend(velocity_grids)
            
            energy_grid = self._create_energy_grid(zarr_store, frame, active_sources)
            if energy_grid:
                vdb_grids.append(energy_grid)
            
            # Save comprehensive scene file
            filename = f"{self.grid_name_prefix}_static_scene.vdb"
            filepath = os.path.join(self.export_path, filename)
            
            vdb.write(filepath, vdb_grids)
            print(f"Static scene exported: {filepath}")
            
        except Exception as e:
            print(f"Error exporting static scene: {e}")


class OptimizedVDBExporter(OpenVDBExporter):
    """Optimized OpenVDB exporter with compression and level set support"""
    
    def __init__(self, voxel_config):
        super().__init__(voxel_config)
        self.use_compression = True
        self.compression_level = 4  # 0-9, higher = better compression
        self.use_level_sets = True
    
    def _create_optimized_grid(self, data: np.ndarray, name: str) -> vdb.FloatGrid:
        """Create optimized OpenVDB grid with compression"""
        grid = vdb.FloatGrid()
        grid.name = name
        grid.copyFromArray(data)
        grid.transform = self._create_transform()
        
        # Apply compression settings
        if self.use_compression:
            grid.compression = self.compression
        
        return grid
    
    def create_level_set_grid(self, threshold: float = 0.1) -> vdb.FloatGrid:
        """
        Create level set grid for isosurface extraction.
        
        Args:
            threshold: Threshold value for level set
        
        Returns:
            Level set grid
        """
        # This would create a level set from your simulation data
        # For now, return a placeholder
        level_set = vdb.FloatGrid()
        level_set.name = "level_set"
        
        # In practice, you'd convert your simulation data to a level set
        # for efficient isosurface extraction in rendering software
        
        return level_set
    
    def export_for_rendering(self, zarr_store, frame: int, render_preset: str = "default"):
        """
        Export data optimized for specific rendering applications.
        
        Args:
            zarr_store: Zarr store with simulation data
            frame: Frame to export
            render_preset: Rendering preset ('default', 'volume', 'isosurface')
        """
        try:
            vdb_grids = []
            
            if render_preset == "volume":
                # Optimize for volume rendering
                pressure_grid = self._create_pressure_grid(zarr_store, frame, 
                                                         self._get_active_sources(zarr_store, frame))
                if pressure_grid:
                    pressure_grid.insertMeta("render_type", "volume")
                    vdb_grids.append(pressure_grid)
            
            elif render_preset == "isosurface":
                # Optimize for isosurface rendering
                energy_grid = self._create_energy_grid(zarr_store, frame,
                                                     self._get_active_sources(zarr_store, frame))
                if energy_grid:
                    energy_grid.insertMeta("render_type", "isosurface")
                    energy_grid.insertMeta("isosurface_threshold", 0.01)
                    vdb_grids.append(energy_grid)
            
            else:  # default
                vdb_grids = self._export_acoustic_fields(zarr_store, frame, vdb.FloatGrid())
            
            # Save with render-specific settings
            filename = f"{self.grid_name_prefix}_render_{render_preset}_{frame:06d}.vdb"
            filepath = os.path.join(self.export_path, filename)
            
            vdb.write(filepath, vdb_grids)
            print(f"Render-optimized export: {filepath}")
            
        except Exception as e:
            print(f"Error in render-optimized export: {e}")

