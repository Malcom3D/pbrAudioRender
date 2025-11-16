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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from core.entity_manager import EntityManager
from lib.functions import _get_position, _world_to_grid, _cartesian_to_spherical
from lib.interpolator import FrequencyInterpolator

@dataclass
class OmnidirectionalOutput:
    """Omnidirectional microphone output with frequency-dependent processing"""
    entity_manager: EntityManager
    idx: int

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.output_config = None
        
        for output_config in config.outputs:
            if output_config.idx == self.idx:
                self.output_config = output_config
        
        self.shape = config.acoustic_domain.shape
        self.voxel_size = config.acoustic_domain.voxel_size
        self.grid_geometry = config.acoustic_domain.geometry
        
        # Frequency bands for processing
        frequency_bands = self.entity_manager.get('frequency_bands')
        self.bands = frequency_bands.get_bands()

    def get_recording_positions(self) -> List[Tuple[float, float, float]]:
        """Get positions where this output should record"""
        positions = []
        
        if self.output_config.position_file:
            try:
                position_data = np.load(self.output_config.position_file)
                for pos in position_data:
                    if len(pos) >= 3:
                        positions.append(tuple(pos[:3]))
            except:
                print(f"Could not load positions from {self.output_config.position_file}")
        
        if not positions and hasattr(self.output_config.geometry, 'position'):
            positions.append(tuple(self.output_config.geometry['position']))
        
        return positions

    def process_audio(self) -> Dict[str, np.ndarray]:
        """
        Process audio for omnidirectional microphone across all frequency bands and layers
        
        Returns:
            Dictionary with 'frequencies' and 'pressures' arrays
        """
        current_frame = self.entity_manager.get('frames').get()
        positions = self.get_recording_positions()
        
        if not positions:
            return {'frequencies': np.array([]), 'pressures': np.array([])}
        
        # Get all wave propagators
        wave_propagators = self.entity_manager.get('wave_propagators')
        
        # Initialize frequency-pressure arrays
        all_frequencies = []
        all_pressures = []
        
        for position in positions:
            grid_pos = _world_to_grid(self.voxel_size, self.grid_geometry, position)
            
            if not self._is_in_bounds(grid_pos):
                continue
                
            # Process each frequency band
            for band_idx, (low_freq, high_freq) in enumerate(self.bands):
                center_freq = np.sqrt(low_freq * high_freq)
                
                # Get interpolated pressure from all layers
                total_pressure = self._get_interpolated_pressure(
                    position, grid_pos, band_idx, center_freq
                )
                
                # Apply frequency response if available
                if hasattr(self.output_config, 'spatial_freq_response'):
                    magnitude_coeff = self.output_config.spatial_freq_response.get_avg_magnitude(
                        0, 0, low_freq, high_freq  # Omnidirectional - no directionality
                    )
                    total_pressure *= magnitude_coeff
                
                # Apply calibration if available
                if hasattr(self.output_config, 'calibration'):
                    cal_coeff = self.output_config.calibration.get_avg_magnitude(
                        0, 0, low_freq, high_freq
                    )
                    total_pressure *= cal_coeff
                
                all_frequencies.append(center_freq)
                all_pressures.append(total_pressure)
        
        return {
            'frequencies': np.array(all_frequencies),
            'pressures': np.array(all_pressures)
        }

    def _get_interpolated_pressure(self, world_pos: Tuple[float, float, float], 
                                 grid_pos: Tuple[int, int, int], 
                                 band_idx: int, frequency: float) -> float:
        """
        Get sub-voxel interpolated pressure from all layers
        """
        total_pressure = 0.0
        wave_propagators = self.entity_manager.get('wave_propagators')
        
        for wp_idx, wave_propagator in wave_propagators.items():
            layer_manager = wave_propagator.layer_manager
            
            # Get pressure from FDTD layers
            fdtd_layers = self._get_fdtd_pressure(layer_manager, band_idx, world_pos, grid_pos)
            total_pressure += fdtd_layers
            
            # Get pressure from reflection layers
            reflection_pressure = self._get_reflection_pressure(layer_manager, band_idx, world_pos, grid_pos)
            total_pressure += reflection_pressure
            
            # Get pressure from scattering layers  
            scattering_pressure = self._get_scattering_pressure(layer_manager, band_idx, world_pos, grid_pos)
            total_pressure += scattering_pressure
        
        return total_pressure

    def _get_fdtd_pressure(self, layer_manager, band_idx: int, 
                          world_pos: Tuple[float, float, float],
                          grid_pos: Tuple[int, int, int]) -> float:
        """Get interpolated pressure from FDTD layers"""
        try:
            layer = layer_manager.get_layer('fdtd', band_idx)
            if layer is None:
                return 0.0
                
            # Trilinear interpolation for sub-voxel accuracy
            pressure = self._trilinear_interpolate(layer, 'pressure', world_pos, grid_pos)
            return pressure
        except:
            return 0.0

    def _get_reflection_pressure(self, layer_manager, band_idx: int,
                               world_pos: Tuple[float, float, float],
                               grid_pos: Tuple[int, int, int]) -> float:
        """Get interpolated pressure from reflection layers"""
        total_reflection = 0.0
        reflection_count = layer_manager.len_by_name('reflection')
        
        for ref_idx in range(reflection_count):
            try:
                layer = layer_manager.get_layer('reflection', ref_idx)
                if layer is not None:
                    pressure = self._trilinear_interpolate(layer, 'pressure', world_pos, grid_pos)
                    total_reflection += pressure
            except:
                continue
                
        return total_reflection

    def _get_scattering_pressure(self, layer_manager, band_idx: int,
                               world_pos: Tuple[float, float, float],
                               grid_pos: Tuple[int, int, int]) -> float:
        """Get interpolated pressure from scattering layers"""
        total_scattering = 0.0
        scattering_count = layer_manager.len_by_name('scattering')
        
        for scat_idx in range(scattering_count):
            try:
                layer = layer_manager.get_layer('scattering', scat_idx)
                if layer is not None:
                    pressure = self._trilinear_interpolate(layer, 'pressure', world_pos, grid_pos)
                    total_scattering += pressure
            except:
                continue
                
        return total_scattering

    def _trilinear_interpolate(self, layer, field_type: str,
                             world_pos: Tuple[float, float, float],
                             grid_pos: Tuple[int, int, int]) -> float:
        """
        Trilinear interpolation for sub-voxel accuracy
        """
        i, j, k = grid_pos
        dx, dy, dz = self.voxel_size
        
        # Calculate fractional positions within the voxel
        base_x = self.grid_geometry[0][0] + i * dx
        base_y = self.grid_geometry[0][1] + j * dy  
        base_z = self.grid_geometry[0][2] + k * dz
        
        x_frac = (world_pos[0] - base_x) / dx
        y_frac = (world_pos[1] - base_y) / dy
        z_frac = (world_pos[2] - base_z) / dz
        
        # Clamp fractions to [0, 1]
        x_frac = max(0, min(1, x_frac))
        y_frac = max(0, min(1, y_frac))
        z_frac = max(0, min(1, z_frac))
        
        # Get values at surrounding grid points
        values = np.zeros((2, 2, 2))
        
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    ni, nj, nk = i + di, j + dj, k + dk
                    if self._is_in_bounds((ni, nj, nk)):
                        try:
                            if field_type == 'pressure':
                                values[di, dj, dk] = layer.field[ni, nj, nk].pressure
                            elif field_type == 'vx':
                                values[di, dj, dk] = layer.field[ni, nj, nk].velocity.vx
                            elif field_type == 'vy':
                                values[di, dj, dk] = layer.field[ni, nj, nk].velocity.vy
                            elif field_type == 'vz':
                                values[di, dj, dk] = layer.field[ni, nj, nk].velocity.vz
                        except:
                            values[di, dj, dk] = 0.0
        
        # Perform trilinear interpolation
        c00 = values[0,0,0] * (1 - x_frac) + values[1,0,0] * x_frac
        c01 = values[0,0,1] * (1 - x_frac) + values[1,0,1] * x_frac
        c10 = values[0,1,0] * (1 - x_frac) + values[1,1,0] * x_frac
        c11 = values[0,1,1] * (1 - x_frac) + values[1,1,1] * x_frac
        
        c0 = c00 * (1 - y_frac) + c10 * y_frac
        c1 = c01 * (1 - y_frac) + c11 * y_frac
        
        interpolated_value = c0 * (1 - z_frac) + c1 * z_frac
        
        return interpolated_value

    def _is_in_bounds(self, grid_pos: Tuple[int, int, int]) -> bool:
        """Check if grid position is within bounds"""
        i, j, k = grid_pos
        return (0 <= i < self.shape[0] and 
                0 <= j < self.shape[1] and 
                0 <= k < self.shape[2])

    def get_directivity(self, azimuth: float, elevation: float,
                       frequency: Optional[float] = None) -> float:
        """Omnidirectional has uniform directivity"""
        return 1.0
