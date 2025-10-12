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
import zarr
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import soundfile as sf
import json

from ..outputs import create_output
from ..lib.calibration import OutputCalibration


@dataclass
class RecordingConfig:
    """Configuration for audio recording"""
    output_format: str = "wav"
    normalize: bool = True
    peak_level: float = -3.0
    dither: bool = True
    bit_depth: int = 32


class AudioRecorder:
    """Manages audio recording at output positions"""
    
    def __init__(self, outputs_config, voxel_grid_config):
        self.outputs_config = outputs_config
        self.voxel_config = voxel_grid_config
        self.recording_config = RecordingConfig()
        
        # Initialize outputs
        self.outputs = self._initialize_outputs()
        
        # Recording buffers
        self.recording_buffers = {}
        self.current_frame = 0
        
        # Create output directory
        os.makedirs("./exports/audio/", exist_ok=True)
    
    def _initialize_outputs(self) -> Dict[int, Any]:
        """Initialize all output recorders"""
        outputs = {}
        
        for output_config in self.outputs_config:
            output = create_output(output_config)
            outputs[output_config.idx] = output
            
            # Initialize recording buffer
            max_frames = self.voxel_config.sample_frame_limit or 44100
            self.recording_buffers[output_config.idx] = {
                'pressure': np.zeros(max_frames, dtype=np.float32),
                'velocity_x': np.zeros(max_frames, dtype=np.float32),
                'velocity_y': np.zeros(max_frames, dtype=np.float32),
                'velocity_z': np.zeros(max_frames, dtype=np.float32)
            }
        
        return outputs
    
    def record(self, frame: int, zarr_store):
        """
        Record audio at all output positions for current frame
        """
        self.current_frame = frame
        
        for output_idx, output in self.outputs.items():
            # Get output position for current frame
            output.update_position(frame, self._load_output_positions(output_idx))
            
            # Record pressure and velocity at output position
            recorded_data = self._record_at_position(output, frame, zarr_store)
            
            # Store in recording buffer
            buffer = self.recording_buffers[output_idx]
            if frame < len(buffer['pressure']):
                buffer['pressure'][frame] = recorded_data['pressure']
                buffer['velocity_x'][frame] = recorded_data['velocity_x']
                buffer['velocity_y'][frame] = recorded_data['velocity_y']
                buffer['velocity_z'][frame] = recorded_data['velocity_z']
    
    def _record_at_position(self, output, frame: int, zarr_store) -> Dict[str, float]:
        """
        Record pressure and velocity at specific output position
        """
        # Get voxel position of output
        voxel_pos = self._world_to_voxel(output.position)
        
        if not self._is_in_bounds(voxel_pos):
            return {'pressure': 0.0, 'velocity_x': 0.0, 'velocity_y': 0.0, 'velocity_z': 0.0}
        
        # Sum contributions from all sources
        total_pressure = 0.0
        total_velocity_x = 0.0
        total_velocity_y = 0.0
        total_velocity_z = 0.0
        
        for source_config in self.outputs_config:
            source_idx = source_config.idx
            
            # Get layer manager data for this source
            layer_data = zarr_store.get_layer_manager_data(source_idx, frame)
            
            if layer_data:
                i, j, k = voxel_pos
                
                # Apply trilinear interpolation for sub-voxel accuracy
                pressure = self._trilinear_interpolate(layer_data['pressure'], voxel_pos)
                vx = self._trilinear_interpolate(layer_data['velocity_x'], voxel_pos)
                vy = self._trilinear_interpolate(layer_data['velocity_y'], voxel_pos)
                vz = self._trilinear_interpolate(layer_data['velocity_z'], voxel_pos)
                
                # Calculate direction from source to output
                source_direction = self._calculate_source_direction(output, source_idx, frame)
                
                # Apply output directivity and frequency response
                freq = 1000.0  # Placeholder - should come from FDTD frequency analysis
                pressure_recorded = output.record_pressure(pressure, source_direction, freq)
                velocity_recorded = output.record_velocity(np.array([vx, vy, vz]), source_direction, freq)
                
                total_pressure += pressure_recorded
                total_velocity_x += velocity_recorded[0]
                total_velocity_y += velocity_recorded[1]
                total_velocity_z += velocity_recorded[2]
        
        return {
            'pressure': total_pressure,
            'velocity_x': total_velocity_x,
            'velocity_y': total_velocity_y,
            'velocity_z': total_velocity_z
        }
    
    def _world_to_voxel(self, world_pos: np.ndarray) -> Tuple[float, float, float]:
        """Convert world coordinates to voxel coordinates with sub-voxel precision"""
        return (
            world_pos[0] / self.voxel_config.voxel_size,
            world_pos[1] / self.voxel_config.voxel_size,
            world_pos[2] / self.voxel_config.voxel_size
        )
    
    def _is_in_bounds(self, voxel_pos: Tuple[float, float, float]) -> bool:
        """Check if voxel coordinates are within grid bounds"""
        i, j, k = voxel_pos
        shape = self.voxel_config.shape
        
        return (0 <= i < shape[0] and 
                0 <= j < shape[1] and 
                0 <= k < shape[2])
    
    def _trilinear_interpolate(self, field: np.ndarray, 
                             position: Tuple[float, float, float]) -> float:
        """Perform trilinear interpolation at sub-voxel position"""
        i, j, k = position
        
        # Get integer coordinates
        i0, j0, k0 = int(np.floor(i)), int(np.floor(j)), int(np.floor(k))
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        
        # Check bounds
        shape = field.shape
        if (i0 < 0 or i1 >= shape[0] or 
            j0 < 0 or j1 >= shape[1] or 
            k0 < 0 or k1 >= shape[2]):
            return 0.0
        
        # Calculate interpolation weights
        di, dj, dk = i - i0, j - j0, k - k0
        
        # Trilinear interpolation
        c00 = field[i0, j0, k0] * (1 - di) + field[i1, j0, k0] * di
        c01 = field[i0, j0, k1] * (1 - di) + field[i1, j0, k1] * di
        c10 = field[i0, j1, k0] * (1 - di) + field[i1, j1, k0] * di
        c11 = field[i0, j1, k1] * (1 - di) + field[i1, j1, k1] * di
        
        c0 = c00 * (1 - dj) + c10 * dj
        c1 = c01 * (1 - dj) + c11 * dj
        
        value = c0 * (1 - dk) + c1 * dk
        
        return value
    
    def _calculate_source_direction(self, output, source_idx: int, 
                                  frame: int) -> Tuple[float, float]:
        """Calculate direction from output to source"""
        # This would use source position data
        # For now, return a default direction
        return (0.0, 0.0)  # azimuth, elevation
    
    def _load_output_positions(self, output_idx: int) -> np.ndarray:
        """Load position data for output"""
        # This would load from the output's position file
        # For now, return default positions
        max_frames = self.voxel_config.sample_frame_limit or 44100
        return np.zeros((max_frames, 7))  # x,y,z + quaternion
    
    def save_recordings(self):
        """Save all recorded audio to files"""
        for output_idx, output in self.outputs.items():
            buffer = self.recording_buffers[output_idx]
            
            # Get valid frames (up to current frame)
            valid_frames = slice(0, min(self.current_frame + 1, len(buffer['pressure'])))
            
            # Extract audio data
            pressure_audio = buffer['pressure'][valid_frames]
            velocity_audio = np.stack([
                buffer['velocity_x'][valid_frames],
                buffer['velocity_y'][valid_frames],
                buffer['velocity_z'][valid_frames]
            ], axis=-1)
            
            # Apply calibration
            pressure_audio = output.apply_calibration(pressure_audio)
            velocity_audio = output.apply_calibration(velocity_audio)
            
            # Save pressure audio
            pressure_file = f"./exports/audio/output_{output_idx}_pressure.wav"
            self._save_audio_file(pressure_audio, pressure_file)
            
            # Save velocity audio (multichannel)
            velocity_file = f"./exports/audio/output_{output_idx}_velocity.wav"
            self._save_audio_file(velocity_audio, velocity_file)
            
            # Save metadata
            self._save_recording_metadata(output_idx, pressure_file, velocity_file)
    
    def _save_audio_file(self(self, audio_data: np.ndarray, file_path: str):
        """Save audio data to file"""
        try:
            # Normalize if configured
            if self.recording_config.normalize:
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val * (10 ** (self.recording_config.peak_level / 20))
            
            # Convert bit depth
            if self.recording_config.bit_depth == 16:
                audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
                subtype = 'PCM_16'
            elif self.recording_config.bit_depth == 24:
                audio_data = np.clip(audio_data * 8388607, -8388608, 8388607).astype(np.int32)
                subtype = 'PCM_24'
            else:  # 32-bit float
                subtype = 'FLOAT'
            
            sf.write(file_path, audio_data, self.voxel_config.sample_rate, subtype=subtype)
            
        except Exception as e:
            print(f"Error saving audio file {file_path}: {e}")
    
    def _save_recording_metadata(self, output_idx: int, pressure_file: str, velocity_file: str):
        """Save recording metadata"""
        metadata = {
            'output_idx': output_idx,
            'output_name': self.outputs[output_idx].name,
            'pressure_file': pressure_file,
            'velocity_file': velocity_file,
            'sample_rate': self.voxel_config.sample_rate,
            'bit_depth': self.recording_config.bit_depth,
            'total_frames': self.current_frame + 1,
            'recording_date': np.datetime64('now').astype(str)
        }
        
        metadata_file = f"./exports/audio/output_{output_idx}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_recording_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for all recordings"""
        stats = {}
        
        for output_idx, buffer in self.recording_buffers.items():
            valid_frames = slice(0, min(self.current_frame + 1, len(buffer['pressure'])))
            
            pressure_data = buffer['pressure'][valid_frames]
            velocity_data = np.stack([
                buffer['velocity_x'][valid_frames],
                buffer['velocity_y'][valid_frames],
                buffer['velocity_z'][valid_frames]
            ], axis=-1)
            
            stats[output_idx] = {
                'pressure_rms': np.sqrt(np.mean(pressure_data ** 2)),
                'pressure_peak': np.max(np.abs(pressure_data)),
                'velocity_rms': np.sqrt(np.mean(velocity_data ** 2)),
                'velocity_peak': np.max(np.abs(velocity_data)),
                'total_frames': len(pressure_data),
                'duration_seconds': len(pressure_data) / self.voxel_config.sample_rate
            }
        
        return stats

