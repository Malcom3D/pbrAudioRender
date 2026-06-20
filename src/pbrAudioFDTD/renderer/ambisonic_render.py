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
import json
from pbrAudioCommon import np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import soundfile as sf
from pathlib import Path
from scipy import signal
import warnings

from ..core.entity_manager import EntityManager
from ..lib.functions import _cartesian_to_spherical

@dataclass
class AmbisonicRender:
    """Render ambisonic audio from recorded microphone data"""
    entity_manager: EntityManager
    
    def __post_init__(self):
        self.config = self.entity_manager.get('config')
        self.ambisonic_config = self.config.ambisonic_render
        
        # Create output directory
        output_dir = Path(self.ambisonic_config.path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all ambisonic outputs from configuration
        self.ambisonic_outputs = []
        for output_config in self.config.outputs:
            if output_config.type == 'ambisonic':
                self.ambisonic_outputs.append(output_config)
        
        # Pre-calculate spherical harmonics for all microphone positions
        self.spherical_harmonics_cache = {}
        
    def render(self):
        """Render ambisonic audio for all ambisonic outputs"""
        print("Starting ambisonic rendering...")
        
        for output_config in self.ambisonic_outputs:
            print(f"Processing ambisonic output: {output_config.name}")
            
            # Load microphone data from npz file
            npz_file = f"{output_config.render_output_path}/{output_config.idx}.npz"
            
            if not os.path.exists(npz_file):
                print(f"Warning: No recorded data found for {output_config.name}")
                continue
            
            try:
                # Load recorded data
                recorded_data = np.load(npz_file)
                mic_signals = recorded_data[recorded_data.files[0]]
                
                # Convert complex signals to real-valued audio
                mic_signals_real = np.real(mic_signals)
                
                # Load spatial arrangement configuration
                with open(output_config.spatial_arrangement_file, 'r') as f:
                    spatial_config = json.load(f)
                
                # Get ambisonic order from config or spatial arrangement
                ambisonic_order = self._get_ambisonic_order(output_config, spatial_config)
                
                # Calculate encoding matrix
                encoding_matrix = self._calculate_encoding_matrix(
                    spatial_config, 
                    ambisonic_order,
                    output_config
                )
                
                # Encode to ambisonic B-format
                b_format_signals = self._encode_to_bformat(
                    mic_signals_real, 
                    encoding_matrix,
                    ambisonic_order
                )
                
                # Apply N3D normalization
                b_format_signals = self._apply_n3d_normalization(
                    b_format_signals, 
                    ambisonic_order
                )
                
                # Resample if needed
                b_format_signals = self._resample_audio(
                    b_format_signals,
                    self.config.acoustic_domain.sample_rate,
                    self.ambisonic_config.sample_rate
                )
                
                # Convert bit depth
                b_format_signals = self._convert_bit_depth(
                    b_format_signals,
                    self.ambisonic_config.bit_depth
                )
                
                # Save to file
                self._save_ambisonic_file(
                    b_format_signals,
                    output_config,
                    ambisonic_order
                )
                
                print(f"Successfully rendered ambisonic audio for {output_config.name}")
                
            except Exception as e:
                print(f"Error rendering ambisonic audio for {output_config.name}: {e}")
                import traceback
                traceback.print_exc()
    
    def _get_ambisonic_order(self, output_config, spatial_config) -> int:
        """Get ambisonic order from configuration"""
        # Priority: output config > spatial config > default
        if hasattr(output_config, 'ambisonic_order') and output_config.ambisonic_order is not None:
            return output_config.ambisonic_order
        elif 'order_supported' in spatial_config:
            return spatial_config['order_supported']
        else:
            return self.ambisonic_config.order
    
    def _calculate_encoding_matrix(self, spatial_config: Dict, 
                                 ambisonic_order: int,
                                 output_config) -> np.ndarray:
        """
        Calculate encoding matrix for ambisonic rendering.
        
        For omnidirectional microphones in symmetric arrays, we can use
        the pseudo-inverse of the spherical harmonics matrix.
        """
        mic_positions = []
        mic_types = []
        
        # Extract microphone positions and types
        for mic_config in spatial_config['outputs']:
            position = np.array(mic_config['position'])
            mic_type = mic_config['type']
            mic_positions.append(position)
            mic_types.append(mic_type)
        
        mic_positions = np.array(mic_positions)
        
        # Calculate spherical coordinates for each microphone
        spherical_coords = []
        for pos in mic_positions:
            x, y, z = pos
            azimuth, elevation, _ = _cartesian_to_spherical(x, y, z)
            spherical_coords.append((azimuth, elevation))
        
        # Calculate number of ambisonic channels
        num_ambisonic_channels = (ambisonic_order + 1) ** 2
        
        # Build spherical harmonics matrix
        Y_matrix = np.zeros((len(mic_positions), num_ambisonic_channels))
        
        for i, (azimuth, elevation) in enumerate(spherical_coords):
            # Calculate spherical harmonics for this direction
            Y = self._calculate_spherical_harmonics(
                azimuth, elevation, ambisonic_order
            )
            Y_matrix[i, :] = Y
        
        # Calculate encoding matrix (pseudo-inverse for decoding matrix)
        # For encoding, we need the inverse of the sampling matrix
        if len(mic_positions) >= num_ambisonic_channels:
            # Overdetermined system, use least squares
            encoding_matrix = np.linalg.pinv(Y_matrix)
        else:
            # Underdetermined system, use transpose (simplified)
            encoding_matrix = Y_matrix.T / np.sum(Y_matrix**2, axis=1)
        
        # Apply microphone directivity patterns
        encoding_matrix = self._apply_microphone_directivity(
            encoding_matrix, mic_types, spherical_coords, ambisonic_order
        )
        
        # Apply normalization from spatial config if specified
        if 'normalization' in spatial_config:
            if spatial_config['normalization'].upper() == 'SN3D':
                encoding_matrix = self._apply_sn3d_normalization(
                    encoding_matrix, ambisonic_order
                )
        
        return encoding_matrix
    
    def _calculate_spherical_harmonics(self, azimuth: float, elevation: float, 
                                     order: int) -> np.ndarray:
        """
        Calculate real spherical harmonics (ACN convention) for given direction.
        
        ACN (Ambisonic Channel Number) convention:
        Channel 0: W (omnidirectional)
        Channel 1: Y (front-back)
        Channel 2: Z (up-down)
        Channel 3: X (left-right)
        etc.
        """
        # Check cache first
        cache_key = (azimuth, elevation, order)
        if cache_key in self.spherical_harmonics_cache:
            return self.spherical_harmonics_cache[cache_key]
        
        num_channels = (order + 1) ** 2
        Y = np.zeros(num_channels)
        
        # Convert to spherical coordinates (physics convention)
        # azimuth: 0 = front, π/2 = left, π = back, 3π/2 = right
        # elevation: 0 = horizontal, π/2 = up, -π/2 = down
        theta = np.pi/2 - elevation  # colatitude (0 at north pole, π at south pole)
        phi = azimuth  # azimuth
        
        # Pre-calculate trigonometric functions
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        # Channel 0: W (omnidirectional)
        Y[0] = 1.0  # N3D normalization already applied
        
        if order >= 1:
            # First order
            Y[1] = sin_theta * cos_phi  # Y
            Y[2] = cos_theta            # Z
            Y[3] = sin_theta * sin_phi  # X
        
        if order >= 2:
            # Second order
            Y[4] = np.sqrt(3)/2 * sin_theta**2 * np.cos(2*phi)  # V
            Y[5] = np.sqrt(3) * sin_theta * cos_theta * cos_phi  # T
            Y[6] = (3*cos_theta**2 - 1)/2                       # R
            Y[7] = np.sqrt(3) * sin_theta * cos_theta * sin_phi  # S
            Y[8] = np.sqrt(3)/2 * sin_theta**2 * np.sin(2*phi)  # U
        
        if order >= 3:
            # Third order
            Y[9] = np.sqrt(5)/2 * sin_theta**3 * np.cos(3*phi)   # Q
            Y[10] = np.sqrt(30)/4 * sin_theta**2 * cos_theta * np.cos(2*phi)  # O
            Y[11] = np.sqrt(3)/2 * sin_theta * (5*cos_theta**2 - 1) * np.cos(phi)  # M
            Y[12] = (5*cos_theta**3 - 3*cos_theta)/2            # K
            Y[13] = np.sqrt(3)/2 * sin_theta * (5*cos_theta**2 - 1) * np.sin(phi)  # L
            Y[14] = np.sqrt(30)/4 * sin_theta**2 * cos_theta * np.sin(2*phi)  # N
            Y[15] = np.sqrt(5)/2 * sin_theta**3 * np.sin(3*phi)   # P
        
        # Cache the result
        self.spherical_harmonics_cache[cache_key] = Y
        
        return Y
    
    def _apply_microphone_directivity(self, encoding_matrix: np.ndarray,
                                    mic_types: List[str],
                                    spherical_coords: List[Tuple[float, float]],
                                    order: int) -> np.ndarray:
        """Apply microphone directivity patterns to encoding matrix"""
        for i, mic_type in enumerate(mic_types):
            azimuth, elevation = spherical_coords[i]
            
            # Calculate directivity gain for this microphone type
            if mic_type == 'omnidirectional':
                gain = 1.0
            elif mic_type == 'cardioid':
                # Cardioid: 0.5 * (1 + cos(θ))
                # θ is angle between sound incidence and microphone forward direction
                # Assuming microphone points in radial direction
                gain = 0.5 * (1 + np.cos(azimuth))
            elif mic_type == 'figure8':
                # Figure-8: cos(θ)
                gain = np.cos(azimuth)
            elif mic_type == 'hypercardioid':
                # Hypercardioid: 0.25 * (1 + 3 * cos(θ))
                gain = 0.25 * (1 + 3 * np.cos(azimuth))
            else:
                gain = 1.0
            
            # Apply gain to encoding matrix row
            encoding_matrix[:, i] *= gain
        
        return encoding_matrix
    
    def _apply_sn3d_normalization(self, encoding_matrix: np.ndarray, 
                                order: int) -> np.ndarray:
        """Apply SN3D (Schmidt semi-normalized) normalization"""
        # SN3D normalization factors
        sn3d_factors = [1.0]  # W channel
        
        if order >= 1:
            sn3d_factors.extend([1/np.sqrt(3)] * 3)  # First order
        
        if order >= 2:
            sn3d_factors.extend([1/np.sqrt(5)] * 5)  # Second order
        
        if order >= 3:
            sn3d_factors.extend([1/np.sqrt(7)] * 7)  # Third order
        
        # Apply normalization
        for i, factor in enumerate(sn3d_factors):
            if i < encoding_matrix.shape[0]:
                encoding_matrix[i, :] *= factor
        
        return encoding_matrix
    
    def _apply_n3d_normalization(self, b_format_signals: np.ndarray,
                               order: int) -> np.ndarray:
        """Apply N3D (full 3D) normalization to B-format signals"""
        # N3D normalization factors (already included in spherical harmonics)
        # This is mostly for verification and final scaling
        
        # Reference: W channel should have RMS similar to input
        # Scale all channels to maintain proper energy distribution
        
        # Calculate energy in W channel
        w_energy = np.mean(b_format_signals[0, :] ** 2)
        
        if w_energy > 0:
            # Scale to maintain reasonable levels
            scale_factor = 1.0 / np.sqrt(w_energy)
            b_format_signals *= scale_factor
        
        return b_format_signals
    
    def _encode_to_bformat(self, mic_signals: np.ndarray,
                         encoding_matrix: np.ndarray,
                         order: int) -> np.ndarray:
        """Encode microphone signals to B-format"""
        num_frames = mic_signals.shape[0]
        num_mics = mic_signals.shape[1] if mic_signals.ndim > 1 else 1
        num_bformat_channels = (order + 1) ** 2
        
        if mic_signals.ndim == 1:
            mic_signals = mic_signals.reshape(-11, 1)
        
        # Encode using matrix multiplication
        b_format = np.zeros((num_bformat_channels, num_frames))
        
        for frame in range(num_frames):
            mic_frame = mic_signals[frame, :]
            b_format[:, frame] = encoding_matrix @ mic_frame
        
        return b_format
    
    def _resample_audio(self, audio_data: np.ndarray,
                       input_sample_rate: int,
                       output_sample_rate: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if input_sample_rate == output_sample_rate:
            return audio_data
        
        # Calculate resampling ratio
        ratio = output_sample_rate / input_sample_rate
        
        # Resample each channel
        num_channels, num_samples = audio_data.shape
        new_num_samples = int(num_samples * ratio)
        
        resampled_data = np.zeros((num_channels, new_num_samples))
        
        for ch in range(num_channels):
            resampled_data[ch, :] = signal.resample(
                audio_data[ch, :],
                new_num_samples
            )
        
        return resampled_data
    
    def _convert_bit_depth(self, audio_data: np.ndarray,
                         target_bit_depth: int) -> np.ndarray:
        """Convert audio to target bit depth"""
        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Convert to target bit depth
        if target_bit_depth == 16:
            # Scale to int16 range
            audio_data = (audio_data * 32767).astype(np.int16)
        elif target_bit_depth == 24:
            # Scale to int24 range (stored in int32)
            audio_data = (audio_data * 8388607).astype(np.int32)
        elif target_bit_depth == 32:
            # Keep as float32
            audio_data = audio_data.astype(np.float32)
        else:
            # Default to float32
            audio_data = audio_data.astype(np.float32)
        
        return audio_data
    
    def _save_ambisonic_file(self, b_format_signals: np.ndarray,
                           output_config,
                           order: int):
        """Save ambisonic audio to file"""
        # Determine file format
        file_format = self.ambisonic_config.file_format.lower()
        
        # Create filename
        filename = f"{output_config.name}_order{order}"
        
        if file_format == 'wav':
            filename += '.wav'
            subtype = 'PCM_16' if self.ambisonic_config.bit_depth == 16 else 'FLOAT'
            self._save_wav_file(b_format_signals, filename, subtype)
        elif file_format == 'bwf':
            filename += '.wav'  # BWF is WAV with metadata
            self._save_bwf_file(b_format_signals, filename, order, output_config)
        else:
            # Default to WAV
            filename += '.wav'
            subtype = 'PCM_16' if self.ambisonic_config.bit_depth == 16 else 'FLOAT'
            self._save_wav_file(b_format_signals, filename, subtype)
    
    def _save_wav_file(self, audio_data: np.ndarray,
                      filename: str,
                      subtype: str):
        """Save as standard WAV file"""
        filepath = os.path.join(self.ambisonic_config.path, filename)
        
        # Transpose for soundfile (channels x samples -> samples x channels)
        audio_data = audio_data.T
        
        sf.write(
            filepath,
            audio_data,
            self.ambisonic_config.sample_rate,
            subtype=subtype
        )
        
        print(f"Saved WAV file: {filepath}")
    
    def _save_bwf_file(self, audio_data: np.ndarray,
                      filename: str,
                      order: int,
                      output_config):
        """Save as Broadcast Wave Format (BWF) with ambisonic metadata"""
        filepath = os.path.join(self.ambisonic_config.path, filename)
        
        # Transpose for soundfile
        audio_data = audio_data.T
        
        # Create BWF metadata
        metadata = {
            'description': f'Ambisonic B-format Order {order}',
            'originator': 'pbrAudioRender',
            'originator_reference': output_config.name,
            'coding_history': f'A{order}',
            'time_reference': 0,
            'version': 1
        }
        
        # Add ambisonic-specific metadata
        metadata['ambisonic_order'] = order
        metadata['normalization'] = 'N3D'
        metadata['channel_names'] = self._get_acn_channel_names(order)
        
        sf.write(
            filepath,
            audio_data,
            self.ambisonic_config.sample_rate,
            subtype='FLOAT',  # BWF typically uses float
            format='WAV',
            endian='FILE',
            metadata=metadata
        )
        
        print(f"Saved BWF file: {filepath}")
    
    def _get_acn_channel_names(self, order: int) -> List[str]:
        """Get ACN channel names for given order"""
        channel_names = ['W']
        
        if order >= 1:
            channel_names.extend(['Y', 'Z', 'X'])
        
        if order >= 2:
            channel_names.extend(['V', 'T', 'R', 'S', 'U'])
        
        if order >= 3:
            channel_names.extend(['Q', 'O', 'M', 'K', 'L', 'N', 'P'])
        
        # Truncate to actual number of channels
        num_channels = (order + 1) ** 2
        return channel_names[:num_channels]
