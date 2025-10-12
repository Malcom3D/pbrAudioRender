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
Ambisonic B-format encoder for 3D acoustic simulation.
Converts multi-channel microphone recordings to Ambisonic B-format (up to 3rd order).
"""

import numpy as np
import soundfile as sf
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from scipy.special import sph_harm


class AmbisonicEncoder:
    """Encodes multi-channel audio to Ambisonic B-format"""
    
    def __init__(self, sample_rate: int, bit_depth: int, outputs_config):
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.outputs_config = outputs_config
        self.ambisonic_order = 3  # Maximum supported order
        self.normalization = "N3D"  # N3D normalization scheme
        self.channel_ordering = "ACN"  # Ambisonic Channel Numbering
        
        # Output directory
        self.export_path = "./exports/ambisonic/"
        Path(self.export_path).mkdir(parents=True, exist_ok=True)
        
        print(f"Ambisonic encoder initialized: order {self.ambisonic_order}, {self.normalization} normalization")
    
    def encode(self):
        """Encode all outputs to Ambisonic B-format"""
        for output_config in self.outputs_config:
            try:
                self._encode_output(output_config)
            except Exception as e:
                print(f"Error encoding output {output_config.idx}: {e}")
    
    def _encode_output(self, output_config):
        """Encode a single output to Ambisonic"""
        # Load spatial arrangement
        spatial_arrangement = self._load_spatial_arrangement(output_config.spatial_arrangement_file)
        if not spatial_arrangement:
            print(f"No spatial arrangement found for output {output_config.idx}")
            return
        
        # Load recorded audio data
        audio_data = self._load_recorded_audio(output_config)
        if audio_data is None:
            print(f"No audio data found for output {output_config.idx}")
            return
        
        # Calculate encoding matrix
        encoding_matrix = self._calculate_encoding_matrix(spatial_arrangement)
        
        # Encode to Ambisonic
        ambisonic_signals = self._encode_audio(audio_data, encoding_matrix, spatial_arrangement)
        
        # Save Ambisonic files
        self._save_ambisonic_files(ambisonic_signals, output_config)
        
        print(f"Encoded output {output_config.idx} to Ambisonic B-format")
    
    def _load_spatial_arrangement(self, arrangement_file: str) -> List[Dict[str, Any]]:
        """Load microphone spatial arrangement from JSON file"""
        try:
            with open(arrangement_file, 'r') as f:
                data = json.load(f)
            return data.get('microphones', [])
        except Exception as e:
            print(f"Error loading spatial arrangement {arrangement_file}: {e}")
            return []
    
    def _load_recorded_audio(self, output_config) -> Optional[Dict[str, np.ndarray]]:
        """Load recorded audio data for output"""
        try:
            audio_data = {}
            
            # Load pressure audio
            pressure_file = f"./exports/audio/output_{output_config.idx}_pressure.wav"
            if Path(pressure_file).exists():
                pressure_data, _ = sf.read(pressure_file)
                audio_data['pressure'] = pressure_data
            
            # Load velocity audio
            velocity_file = f"./exports/audio/output_{output_config.idx}_velocity.wav"
            if Path(velocity_file).exists():
                velocity_data, _ = sf.read(velocity_file)
                audio_data['velocity'] = velocity_data
            
            return audio_data if audio_data else None
            
        except Exception as e:
            print(f"Error loading audio for output {output_config.idx}: {e}")
            return None
    
    def _calculate_encoding_matrix(self, microphones: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate encoding matrix from microphone positions"""
        num_mics = len(microphones)
        num_channels = (self.ambisonic_order + 1) ** 2
        
        encoding_matrix = np.zeros((num_channels, num_mics), dtype=np.float64)
        
        for mic_idx, microphone in enumerate(microphones):
            azimuth = np.deg2rad(microphone.get('azimuth', 0.0))
            elevation = np.deg2rad(microphone.get('elevation', 0.0))
            
            # Calculate spherical harmonics for this direction
            channel_idx = 0
            for order in range(self.ambisonic_order + 1):
                for degree in range(-order, order + 1):
                    # Calculate spherical harmonic
                    harmonic = sph_harm(degree, order, azimuth, elevation)
                    
                    # Apply N3D normalization
                    if self.normalization == "N3D":
                        normalization_factor = np.sqrt((2 * order + 1) / (4 * np.pi))
                        if degree == 0:
                            normalization_factor /= np.sqrt(2)  # W channel special case
                    
                    # Store in encoding matrix
                    encoding_matrix[channel_idx, mic_idx] = (harmonic.real * normalization_factor)
                    channel_idx += 1
        
        return encoding_matrix
    
    def _encode_audio(self, audio_data: Dict[str, np.ndarray], 
                     encoding_matrix: np.ndarray,
                     microphones: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Encode audio data to Ambisonic B-format"""
        num_channels = encoding_matrix.shape[00]
        num_samples = len(audio_data.get('pressure', np.array([])))
        
        # Initialize Ambisonic signals
        ambisonic_signals = {
            'pressure_based': np.zeros((num_samples, num_channels), dtype=np.float32),
            'velocity_based': np.zeros((num_samples, num_channels), dtype=np.float32)
        }
        
        if 'pressure' in audio_data:
            # Pressure-based encoding (conventional approach)
            pressure_data = audio_data['pressure']
            for mic_idx, microphone in enumerate(microphones):
                if mic_idx < len(pressure_data):
                    mic_signal = pressure_data[mic_idx] if pressure_data.ndim > 1 else pressure_data
                    for channel in range(num_channels):
                        ambisonic_signals['pressure_based'][:, channel] += (
                            mic_signal * encoding_matrix[channel, mic_idx]
                        )
        
        if 'velocity' in audio_data:
            # Velocity-based encoding (for higher accuracy)
            velocity_data = audio_data['velocity']
            if velocity_data.ndim == 2 and velocity_data.shape[1] >= 3:
                # Use velocity vector for encoding
                for mic_idx, microphone in enumerate(microphones):
                    if mic_idx < velocity_data.shape[0]:
                        # Get velocity vector for this microphone
                        vx = velocity_data[mic_idx, 0] if velocity_data.ndim > 1 else 00
                        vy = velocity_data[mic_idx, 1] if velocity_data.ndim > 1 else 0
                        vz = velocity_data[mic_idx, 2] if velocity_data.ndim > 1 else 0
                        
                        # Convert velocity to spherical components
                        azimuth = np.deg2rad(microphone.get('azimuth', 0.0))
                        elevation = np.deg2rad(microphone.get('elevation', 0.0))
                        
                        # Encode velocity components
                        # This is a simplified version - full implementation would use
                        # velocity components in spherical harmonic domain
                        for channel in range(num_channels):
                            # Placeholder for velocity encoding
                            velocity_contribution = (vx + vy + vz) * encoding_matrix[channel, mic_idx]
                            ambisonic_signals['velocity_based'][:, channel] += velocity_contribution
        
        return ambisonic_signals
    
    def _save_ambisonic_files(self, ambisonic_signals: Dict[str, np.ndarray], output_config):
        """Save encoded Ambisonic signals to files"""
        output_idx = output_config.idx
        
        for encoding_type, signals in ambisonic_signals.items():
            if signals.size > 0:
                filename = f"ambisonic_output_{output_idx}_{encoding_type}.wav"
                filepath = Path(self.export_path) / filename
                
                # Normalize signals
                max_val = np.max(np.abs(signals))
                if max_val > 0:
                    signals_normalized = signals / max_val * 0.9  # -3 dB headroom
                else:
                    signals_normalized = signals
                
                # Save as WAV file
                sf.write(filepath, signals_normalized, self.sample_rate, 
                        subtype=self._get_subtype())
                
                # Save metadata
                self._save_ambisonic_metadata(output_idx, encoding_type, filepath)
    
    def _get_subtype(self) -> str:
        """Get SoundFile subtype based on bit depth"""
        if self.bit_depth == 16:
            return 'PCM_16'
        elif self.bit_depth == 24:
            return 'PCM_24'
        else:  # 32-bit float
            return 'FLOAT'
    
    def _save_ambisonic_metadata(self, output_idx: int, encoding_type: str, filepath: Path):
        """Save Ambisonic file metadata"""
        metadata = {
            'output_idx': output_idx,
            'encoding_type': encoding_type,
            'ambisonic_order': self.ambisonic_order,
            'normalization': self.normalization,
            'channel_ordering': self.channel_ordering,
            'channel_names': self._get_channel_names(),
            'sample_rate': self.sample_rate,
            'bit_depth': self.bit_depth,
            'file_path': str(filepath),
            'creation_date': np.datetime64('now').astype(str)
        }
        
        metadata_file = filepath.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    
    def _get_channel_names(self) -> List[str]:
        """Get ACN channel names for current Ambisonic order"""
        channel_names = []
        
        for order in range(self.ambisonic_order + 1):
            for degree in range(-order, order + 1):
                if order == 0:
                    channel_names.append('W')
                elif order == 1:
                    if degree == -1: channel_names.append('Y')
                    elif degree == 0: channel_names.append('Z')
                    elif degree == 1: channel_names.append('X')
                else:
                    channel_names.append(f'L{order}_{degree}')
        
        return channel_names
    
    def encode_to_higher_order(self, input_signals: np.ndarray, 
                              input_order: int, target_order: int) -> np.ndarray:
        """
        Encode from lower-order to higher-order Ambisonics.
        
        Args:
            input_signals: Input Ambisonic signals
            input_order: Input Ambisonic order
            target_order: Target Ambisonic order
        
        Returns:
            Higher-order Ambisonic signals
        """
        if target_order <= input_order:
            return input_signals
        
        input_channels = (input_order + 1) ** 2
        target_channels = (target_order + 1) ** 2
        
        if input_signals.shape[1] != input_channels:
            raise ValueError(f"Input signals have {input_signals.shape[1]} channels, "
                           f"expected {input_channels} for order {input_order}")
        
        # Pad with zeros for higher orders
        output_signals = np.zeros((input_signals.shape[0], target_channels), 
                                 dtype=input_signals.dtype)
        output_signals[:, :input_channels] = input_signals
        
        return output_signals
    
    def get_channel_gains(self, azimuth: float, elevation: float) -> np.ndarray:
        """
        Get channel gains for a sound source at specific direction.
        
        Args:
            azimuth: Source azimuth in degrees
            elevation: Source elevation in degrees
        
        Returns:
            Array of channel gains
        """
        azimuth_rad = np.deg2rad(azimuth)
        elevation_rad = np.deg2rad(elevation)
        
        num_channels = (self.ambisonic_order + 1) ** 2
        gains = np.zeros(num_channels, dtype=np.float64)
        
        channel_idx = 0
        for order in range(self.ambisonic_order + 1):
            for degree in range(-order, order + 1):
                harmonic = sph_harm(degree, order, azimuth_rad, elevation_rad)
                
                # Apply N3D normalization
                if self.normalization == "N3D":
                    normalization_factor = np.sqrt((2 * order + 1) / (4 * np.pi))
                    if degree == 0:
                        normalization_factor /= np.sqrt(2)
                
                gains[channel_idx] = (harmonic.real * normalization_factor)
                channel_idx += 1
        
        return gains


class BFormatEncoder(AmbisonicEncoder):
    """Specialized encoder for first-order B-format (W, X, Y, Z)"""
    
    def __init__(self, sample_rate: int, bit_depth: int, outputs_config):
        super().__init__(sample_rate, bit_depth, outputs_config)
        self.ambisonic_order = 1  # Force first-order
    
    def _calculate_encoding_matrix(self, microphones: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate first-order B-format encoding matrix"""
        num_mics = len(microphones)
        encoding_matrix = np.zeros((4, num_mics), dtype=np.float64)  # W, X, Y, Z
        
        for mic_idx, microphone in enumerate(microphones):
            azimuth = np.deg2rad(microphone.get('azimuth', 0.0))
            elevation = np.deg2rad(microphone.get('elevation', 0.0))
            
            # First-order B-format components with N3D normalization
            # W (omnidirectional)
            encoding_matrix[0, mic_idx] = 1.0 / np.sqrt(2)  # N3D normalization
            
            # X (front-back)
            encoding_matrix[1, mic_idx] = np.cos(elevation) * np.cos(azimuth)
            
            # Y (left-right)
            encoding_matrix[2, mic_idx] = np.cos(elevation) * np.sin(azimuth)
            
            # Z (up-down)
            encoding_matrix[3, mic_idx] = np.sin(elevation)
        
        return encoding_matrix


class HigherOrderAmbisonicEncoder(AmbisonicEncoder):
    """Extended encoder with higher-order Ambisonic features"""
    
    def __init__(self, sample_rate: int, bit_depth: int, outputs_config, max_order: int = 3):
        super().__init__(sample_rate, bit_depth, outputs_config)
        self.ambisonic_order = min(max_order, 5)  # Support up to 5th order
    
    def encode_with_head_rotation(self, audio_data: Dict[str, np.ndarray],
                                 rotation_angles: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Encode with dynamic head rotation.
        
        Args:
            audio_data: Input audio data
            rotation_angles: List of (yaw, pitch, roll) angles per frame
        
        Returns:
            Rotated Ambisonic signals
        """
        # This would implement rotation of Ambisonic sound field
        # using rotation matrices for spherical harmonics
        
        # Placeholder implementation
        base_signals = self._encode_audio(audio_data, 
                                         self._calculate_encoding_matrix([]), [])
        
        return base_signals['pressure_based']
    
    def encode_distance_cues(self, audio_data: Dict[str, np.ndarray],
                            distances: List[float]) -> np.ndarray:
        """
        Encode with distance cues for near-field effects.
        
        Args:
            audio_data: Input audio data
            distances: Source distances per channel
        
        Returns:
            Ambisonic signals with distance cues
        """
               # Implement near-field compensation for Ambisonics
        # This adds proximity effects for close sources
        
        # Placeholder implementation
        base_signals = self._encode_audio(audio_data, 
                                         self._calculate_encoding_matrix([]), [])
        
        return base_signals['pressure_based']

