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
import resampy
from pbrAudioCommon import np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import soundfile as sf
from pathlib import Path
from scipy import signal
import warnings

from ..core.entity_manager import EntityManager

@dataclass
class MonoRender:
    """Render mono audio from recorded microphone data"""
    entity_manager: EntityManager
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        
        # Create output directory
        output_dir = Path(config.ambisonic_render.path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    def render(self):
        """Render mono audio for all non-ambisonic outputs"""
        print("Starting mono audio rendering...")
        config = self.entity_manager.get('config')

        # Get all non-ambisonic outputs from configuration
        mono_outputs = []
        for output_config in config.outputs:
            if output_config.type != 'ambisonic':
                mono_outputs.append(output_config)
        
        print(f"Found {len(mono_outputs)} mono outputs to render")
    
        for output_config in mono_outputs:
            print(f"Processing mono output: {output_config.name} (type: {output_config.type})")
            
            # Load microphone data from npz file
            npz_file = f"{output_config.render_output_path}/{output_config.idx}.npz"
            
            if not os.path.exists(npz_file):
                print(f"Warning: No recorded data found for {output_config.name}")
                continue
            
            try:
                # Load recorded data
                recorded_data = np.load(npz_file)
                
                # Handle different array formats
                if len(recorded_data.files) == 0:
                    print(f"Warning: Empty npz file for {output_config.name}")
                    continue
                
                # Get the first array in the npz file
                array_key = recorded_data.files[0]
                mic_signals = recorded_data[array_key]
                
                # Close the npz file
                recorded_data.close()
                
                # Convert complex signals to real-valued audio
                # Take the real part of complex numbers
                if np.iscomplexobj(mic_signals):
                    print(f"Converting complex signals to real values for {output_config.name}")
                    mic_signals_real = np.real(mic_signals)
                else:
                    mic_signals_real = mic_signals
                
                # Ensure we have a 1D array for mono output
                if mic_signals_real.ndim > 1:
                    # If it's a 2D array with shape (n_frames, n_channels), take first channel
                    if mic_signals_real.shape[1] > 1:
                        print(f"Warning: Multi-channel data found for mono output {output_config.name}. Using first channel.")
                        mic_signals_real = mic_signals_real[:, 0]
                    else:
                        # Squeeze to 1D
                        mic_signals_real = mic_signals_real.squeeze()
                
                # Resample if needed
                mic_signals_real = self._resample_audio(
                    mic_signals_real,
                    config.acoustic_domain.sample_rate,
                    config.ambisonic_render.sample_rate
                )
                
                # Normalize to prevent clipping
                mic_signals_real = self._normalize_audio(mic_signals_real)
                
                # Convert bit depth
                mic_signals_real = self._convert_bit_depth(mic_signals_real, config.ambisonic_render.bit_depth)
                
                # Save to file
                self._save_mono_file(mic_signals_real, output_config)
                
                print(f"Successfully rendered mono audio for {output_config.name}")
                
            except Exception as e:
                print(f"Error rendering mono audio for {output_config.name}: {e}")
                import traceback
                traceback.print_exc()
    
    def _resample_audio(self, audio_data: np.ndarray,
                       input_sample_rate: int,
                       output_sample_rate: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        config = self.entity_manager.get('config')
        if input_sample_rate == output_sample_rate:
            return audio_data
        
        # Resample using resampy
        resampled_data = resampy.resample(audio_data, input_sample_rate, output_sample_rate)
        
        return resampled_data
    
    def _normalize_audio(self, audio_data: np.ndarray, 
                        headroom_db: float = -1.0) -> np.ndarray:
        """Normalize audio to prevent clipping with optional headroom"""
        config = self.entity_manager.get('config')
        if len(audio_data) == 0:
            return audio_data
        
        # Find peak value
        peak_value = np.max(np.abs(audio_data))
        
        if peak_value == 0:
            return audio_data
        
        # Calculate normalization factor with headroom
        headroom_linear = 10 ** (headroom_db / 20.0)
        normalization_factor = (1.0 / peak_value) * headroom_linear
        
        # Apply normalization
        normalized_audio = audio_data * normalization_factor
        
        # Clip to safe range (just in case)
        normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
        
        return normalized_audio
    
    def _convert_bit_depth(self, audio_data: np.ndarray, target_bit_depth: int = 32) -> np.ndarray:
        """Convert audio to target bit depth"""
        config = self.entity_manager.get('config')
        # Ensure audio is in float32 format for processing
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize to [-1, 1] range if not already
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
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
        
        return audio_data
    
    def _save_mono_file(self, audio_data: np.ndarray,
                       output_config) -> None:
        """Save mono audio to file"""
        config = self.entity_manager.get('config')
        # Determine file format
        file_format = config.ambisonic_render.file_format.lower()
        
        # Create filename based on output name
        filename = f"{output_config.name}"
        
        # Add file extension based on format
        if file_format == 'wav':
            filename += '.wav'
            subtype = self._get_wav_subtype(config.ambisonic_render.bit_depth)
            self._save_wav_file(audio_data, filename, subtype, output_config)
        elif file_format == 'flac':
            filename += '.flac'
            self._save_flac_file(audio_data, filename, output_config)
        elif file_format == 'aiff':
            filename += '.aiff'
            self._save_aiff_file(audio_data, filename, output_config)
        else:
            # Default to WAV
            filename += '.wav'
            subtype = self._get_wav_subtype(config.ambisonic_render.bit_depth)
            self._save_wav_file(audio_data, filename, subtype, output_config)
    
    def _get_wav_subtype(self, bit_depth: int) -> str:
        """Get appropriate WAV subtype for given bit depth"""
        config = self.entity_manager.get('config')
        if bit_depth == 16:
            return 'PCM_16'
        elif bit_depth == 24:
            return 'PCM_24'
        elif bit_depth == 32:
            # Check if audio is float or int
            return 'FLOAT'  # Default to float for 32-bit
        else:
            return 'FLOAT'  # Default to float
    
    def _save_wav_file(self, audio_data: np.ndarray,
                      filename: str,
                      subtype: str,
                      output_config) -> None:
        """Save as WAV file"""
        config = self.entity_manager.get('config')
        filepath = os.path.join(config.ambisonic_render.path, filename)
        
        # Ensure audio_data is 1D for mono
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()
        
        sf.write(
            filepath,
            audio_data,
            config.ambisonic_render.sample_rate,
            subtype=subtype
        )
        
        print(f"Saved WAV file: {filepath}")
    
    def _save_flac_file(self, audio_data: np.ndarray,
                       filename: str,
                       output_config) -> None:
        """Save as FLAC file"""
        config = self.entity_manager.get('config')
        filepath = os.path.join(config.ambisonic_render.path, filename)
        
        # Ensure audio_data is 1D for mono
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()
        
        # Convert to appropriate format for FLAC (usually 16-bit or 24-bit integer)
        if audio_data.dtype == np.float32:
            # Normalize and convert to int24 for FLAC
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = (audio_data / max_val * 8388607).astype(np.int32)
        
        sf.write(
            filepath,
            audio_data,
            config.ambisonic_render.sample_rate,
            format='FLAC'
        )
        
        print(f"Saved FLAC file: {filepath}")
    
    def _save_aiff_file(self, audio_data: np.ndarray,
                       filename: str,
                       output_config) -> None:
        """Save as AIFF file"""
        config = self.entity_manager.get('config')
        filepath = os.path.join(config.ambisonic_render.path, filename)
        
        # Ensure audio_data is 1D for mono
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()
        
        sf.write(
            filepath,
            audio_data,
            config.ambisonic_render.sample_rate,
            format='AIFF'
        )
        
        print(f"Saved AIFF file: {filepath}")
    
    def render_all(self):
        """Convenience method to render both mono and ambisonic audio"""
        print("Starting complete audio rendering...")
        config = self.entity_manager.get('config')
        
        # Render mono outputs
        self.render()
        
        # Note: Ambisonic rendering would be handled by AmbisonicRender class
        print("Mono rendering complete. Use AmbisonicRender for ambisonic outputs.")

