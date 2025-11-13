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
from outputs.omnidirectional_output import OmnidirectionalOutput

@dataclass
class CardioidOutput(OmnidirectionalOutput):
    """Cardioid microphone output with frequency-dependent processing"""
    
    def process_audio(self) -> Dict[str, np.ndarray]:
        """
        Process audio for cardioid microphone across all frequency bands and layers
        
        Returns:
            Dictionary with 'frequencies' and 'pressures' arrays
        """
        current_frame = self.entity_manager.get('frames').get()
        positions = self.get_recording_positions()
        
        if not positions:
            return {'frequencies': np.array([]), 'pressures': np.array([])}
        
        # Get microphone orientation
        mic_orientation = self._get_orientation(current_frame)
        
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
                
                # Get interpolated pressure and velocity
                pressure = self._get_interpolated_pressure(position, grid_pos, band_idx, center_freq)
                velocity = self._get_interpolated_velocity(position, grid_pos, band_idx, center_freq)
                
                # Calculate velocity component in microphone direction
                velocity_component = np.dot(velocity, mic_orientation)
                
                # Cardioid pattern: 0.5 * (P + V/c * ρ)
                sound_speed = 343.0
                density = 1.2
                cardioid_signal = 0.5 * (pressure + velocity_component / sound_speed * density)
                
                # Apply frequency response if available
                if hasattr(self.output_config, 'spatial_freq_response'):
                    # Get direction-dependent response
                    azimuth, elevation, _ = _cartesian_to_spherical(
                        mic_orientation[0], mic_orientation[1], mic_orientation[2]
                    )
                    magnitude_coeff = self.output_config.spatial_freq_response.get_avg_magnitude(
                        azimuth, elevation,, low_freq, high_freq
                    )
                    cardioid_signal *= magnitude_coeff
                
                # Apply calibration if available
                if hasattr(self.output_config, 'calibration'):
                    cal_coeff = self.output_config.calibration.get_avg_magnitude(
                        0, 0, low_freq, high_freq
                    )
                    cardioid_signal *= cal_coeff
                
                all_frequencies.append(center_freq)
                all_pressures.append(cardioid_signal)
        
        return {
            'frequencies': np.array(all_frequencies),
            'pressures': np.array(all_pressures)
        }

    def _get_interpolated_velocity(self, world_pos: Tuple[float, float, float],
                                 grid_pos: Tuple[int, int, int],
                                 band_idx: int, frequency: float) -> np.ndarray:
        """
        Get sub-voxel interpolated velocity vector from all layers
        """
        total_velocity = np.zeros(3)
        wave_propagators = self.entity_manager.get('wave_propagators')
        
        for wp_idx, wave_propagator in wave_propagators.items():
            layer_manager = wave_propagator.layer_manager
            
            # Get velocity from FDTD layers
            vx = self._get_fdtd_velocity_component(layer_manager, band_idx, world_pos, grid_pos, 'vx')
            vy = self._get_fdtd_velocity_component(layer_manager, band_idx, world_pos, grid_pos, 'vy')
            vz = self._get_fdtd_velocity_component(layer_manager, band_idx, world_pos, grid_pos, 'vz')
            
            total_velocity[0] += vx
            total_velocity[1] += vy
            total_velocity[2] += vz
            
            # Add velocity from reflection and scattering layers (simplified)
            # In practice, you might want more sophisticated handling
        
        return total_velocity

    def _get_fdtd_velocity_component(self, layer_manager, band_idx: int,
                                   world_pos: Tuple[float, float, float],
                                   grid_pos: Tuple[int, int, int],
                                   component: str) -> float:
        """Get interpolated velocity component from FDTD layers"""
        try:
            layer = layer_manager.get_layer('fdtd', band_idx)
            if layer is None:
                return 0.0
                
            velocity_component = self._trilinear_interpolate(layer, component, world_pos, grid_pos)
            return velocity_component
        except:
            return 0.0

    def _get_orientation(self, frame: int) -> np.ndarray:
        """Get microphone orientation for current frame"""
        if hasattr(self.output_config, 'position_file') and self.output_config.position_file:
            try:
                position_data = np.load(self.output_config.position_file)
                if frame < len(position_data) and len(position_data[frame]) >= 6:
                    # Extract orientation from position data (assuming format: x,y,z,ox,oy,oz)
                    return position_data[frame][3:6]
            except:
                pass
        
        # Default orientation (forward)
        return np.array([1.0, 0.0, 0.0])

    def get_directivity(self, azimuth: float, elevation: float,
                       frequency: Optional[float] = None) -> float:
        """Cardioid directivity pattern"""
        return 0.5 * (1 + np.cos(np.deg2rad(azimuth)))
