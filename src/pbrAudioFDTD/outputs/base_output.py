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

import json
from pbrAudioCommon.lib.import_helper import np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.functions import _get_position, _get_rotation, _world_to_grid, _cartesian_to_spherical, _trilinear_interpolate
from ..lib.interpolator import FrequencyInterpolator

@dataclass
class BaseOutput:
    """Base class for all microphone outputs with common functionality"""
    entity_manager: EntityManager
    idx: int
    id: int = None  # microphone id for ambisonic configuration

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.output_config = None

        for output_config in config.outputs:
            if output_config.idx == self.idx:
                self.output_config = output_config

                if not self.id == None:
                    with open(self.output_config.spatial_arrangement_file, 'r') as f:
                        spatial_config = json.load(f)
                        for mic_config in spatial_config['outputs']:
                            if mic_config['id'] == self.id:
                                self.relative_pos = np.array(mic_config['position'])
                                self.relative_rot = np.array(mic_config['rotation'])

    def process_audio(self) -> float:
        """
        Process audio for microphone across all frequency bands and layers
        
        Returns:
            Float: the sum of the higher accuracy encoded pressure and velocity vector 
                   of all layers in the sub voxel interpolated recording position 
                   with sub-voxel accuracy at current frame after applying frequency 
                   response and calibration.
        """
        # Get recording position with sub-voxel accuracy
        recording_position = self._get_recording_position()
        
        # Get interpolated pressure and velocity from all layers
        total_pressure = 0.0
        
        # Get all wave propagators
        wave_propagators = self.entity_manager.get('wave_propagators')
        
        for propagator_idx, wave_propagator in wave_propagators.items():
            layer_manager = wave_propagator.layer_manager
            
            # Process all layers for this propagator
            for layer_idx, layer in layer_manager.layers.items():
                # Get interpolated pressure and velocity at recording position
                pressure = self._get_interpolated_pressure(layer_manager, layer.name, layer.bands_idx, recording_position)
                velocity = self._get_interpolated_velocity(layer_manager, layer.name, layer.bands_idx, recording_position)
                
                # Apply directivity pattern based on microphone type
                azimuth, elevation = self._get_incident_angle(velocity, recording_position)
                directivity = self._get_directivity(azimuth, elevation)
                
                # Combine pressure and velocity contributions
                pressure_contribution = pressure * directivity
                
                # Velocity contribution: project velocity onto microphone direction
                mic_direction = self._get_microphone_direction()
                velocity_contribution = np.dot(velocity, mic_direction) * directivity
                
                # Higher-order accuracy: pressure + velocity component in microphone direction
                raw_value = pressure_contribution + velocity_contribution

                # Apply 3D frequency response and calibration
                calibrated_output = self._apply_calibration(azimuth, elevation, layer.bands_idx, raw_value)

                # Weight the contributions (you can adjust these weights based on microphone type)
                total_pressure += calibrated_output
        
        return total_pressure

    def _get_recording_position(self) -> Tuple[float, float, float]:
        """Get sub-voxel accurate recording position"""
        frames = self.entity_manager.get('frames')
        current_frame = frames.get()
        
        # Get base position
        base_position = _get_position(self.output_config.position_file, current_frame)
        
        # Apply relative position if this is part of an array
        if hasattr(self, 'relative_pos'):
            final_position = base_position + self.relative_pos
        else:
            final_position = base_position
            
        # Convert to grid coordinates with sub-voxel precision
        config = self.entity_manager.get('config')
        voxel_size = config.acoustic_domain.voxel_size
        grid_geometry = config.acoustic_domain.geometry
        
        # Convert to grid coordinates (returns float values for sub-voxel precision)
        grid_position = ((final_position - grid_geometry[0]) / voxel_size)
        
        return tuple(grid_position)

    def _get_interpolated_pressure(self, layer_manager, layer_name: str, bands_idx: int, 
                                 recording_position: Tuple[float, float, float]) -> float:
        """
        Get interpolated pressure from a specific layer with sub-voxel accuracy
        """
        # Get pressure array for this layer and frequency band
        pressure_array = layer_manager.get_array(layer_name, bands_idx, 'pressure')
        
        # Use trilinear interpolation for sub-voxel accuracy
        interpolated_pressure = _trilinear_interpolate(pressure_array, recording_position)
        
        return interpolated_pressure

    def _get_interpolated_velocity(self, layer_manager, layer_name: str, bands_idx: int,
                                 recording_position: Tuple[float, float, float]) -> np.ndarray:
        """
        Get interpolated velocity vectors from a specific layer with sub-voxel accuracy
        """
        # Get velocity arrays for this layer and frequency band
        vx_array = layer_manager.get_array(layer_name, bands_idx, 'vx')
        vy_array = layer_manager.get_array(layer_name, bands_idx, 'vy')
        vz_array = layer_manager.get_array(layer_name, bands_idx, 'vz')
        
        # Use trilinear interpolation for each velocity component
        vx = _trilinear_interpolate(vx_array, recording_position)
        vy = _trilinear_interpolate(vy_array, recording_position)
        vz = _trilinear_interpolate(vz_array, recording_position)
        
        return np.array([vx, vy, vz])

    def _get_incident_angle(self, velocity: np.ndarray, 
                          recording_position: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        Calculate incident angle of sound wave relative to microphone orientation
        """
        # Get microphone orientation
        mic_orientation = self._get_orientation()
        
        # Convert velocity direction to spherical coordinates relative to microphone
        # For simplicity, we''ll use the velocity vector direction
        velocity_magnitude = np.linalg.norm(velocity)
        
        if velocity_magnitude < 1e-10:
            return 0.0, 0.0  # No meaningful direction
            
        velocity_dir = velocity / velocity_magnitude
        
        # Convert to spherical coordinates relative to microphone orientation
        # This is a simplified calculation - you might want to use proper coordinate transformation
        x, y, z = velocity_dir
        azimuth, elevation, _ = _cartesian_to_spherical(x, y, z)
        
        return azimuth, elevation

    def _get_orientation(self) -> Tuple[float, float, float]:
        """Get microphone orientation for current frame"""
        frames = self.entity_manager.get('frames')
        current_frame = frames.get()
        
        rotation = _get_rotation(self.output_config.rotation_file, current_frame)
        if hasattr(self, 'relative_rot'):
            rotation = rotation + self.relative_rot
            
        return rotation

    def _get_microphone_direction(self) -> np.ndarray:
        """
        Get the primary direction vector of the microphone
        For most microphones, this is the forward (z) direction
        """
        orientation = self._get_orientation()
        
        # Convert Euler angles to direction vector
        # Simplified: assume microphone points in local z-direction
        rx, ry, rz = np.radians(orientation)
        
        # Rotation matrix for ZYX convention
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        # Combined rotation
        rotation_matrix = Rz @ Ry @ Rx
        
        # Microphone points in local z-direction
        local_direction = np.array([0, 0, 1])
        world_direction = rotation_matrix @ local_direction
        
        return world_direction

    def _apply_calibration(self, azimuth: float, elevation: float, bands_idx: int, raw_value: float) -> float:
        """
        Apply frequency response and calibration to the raw audio value
        """
        # Get current frequency band information
        frequency_bands = self.entity_manager.get('frequency_bands')
        bands = frequency_bands.get_bands()
        low_freq = bands[bands_idx][0]
        high_freq = bands[bands_idx][1]
    
        # Apply frequency response
        if self.output_config.spatial_freq_response:
            freq_response = self.output_config.spatial_freq_response.get_avg_magnitude(azimuth, elevation, low_freq, high_freq)
            phase_response = self.output_config.spatial_freq_response.get_avg_phase(azimuth, elevation, low_freq, high_freq)
        else:
            freq_response = 1.0
            phase_response = 1.0
                
        # Apply calibration
        if self.output_config.calibration:
            mag_calibration = self.output_config.calibration.get_avg_magnitude(azimuth, elevation, low_freq, high_freq)
            phase_calibration = self.output_config.calibration.get_avg_phase(azimuth, elevation, low_freq, high_freq)
        else:
            mag_calibration = 1.0
            phase_calibration = 1.0

        raw_value = raw_value * freq_response * np.exp(1j * phase_response)
        raw_value = raw_value * mag_calibration * np.exp(1j * phase_calibration)
        return raw_value

    def _get_directivity(self, azimuth: float, elevation: float) -> float:
        """Base directivity pattern - should be overridden by subclasses"""
        return 1.0  # Omnidirectional by default
