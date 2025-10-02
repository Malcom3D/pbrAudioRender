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

import soundfile as sf
import numpy as np
from typing import Tuple, Optional, List
import os

class WAVLoader:
    """Handles loading and validation of mono PCM WAV files for canned sound sources."""
    
    SUPPORTED_SAMPLE_RATES = [44100, 48000, 96000, 192000]
    SUPPORTED_BIT_DEPTHS = [16, 24, 32]
    
    def __init__(self):
        self.loaded_sounds = {}
    
    def load_wav_file(self, file_path: str, normalize: bool = True) -> Tuple[np.ndarray, int]:
        """
        Load a mono PCM WAV file with validation.
        
        Args:
            file_path: Path to the WAV file
            normalize: Whether to normalize audio to [-1, 1] range
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"WAV file not found: {file_path}")
        
        # Load audio data
        try:
            audio_data, sample_rate = sf.read(file_path)
        except Exception as e:
            raise ValueError(f"Error reading WAV file {file_path}: {str(e)}")
        
        # Validate mono
        if len(audio_data.shape) > 1:
            if audio_data.shape[1] > 1:
                print(f"Warning: {file_path} is not mono. Using first channel only.")
                audio_data = audio_data[:, 0]
        
        # Validate sample rate
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            print(f"Warning: Sample rate {sample_rate}Hz is not in supported list {self.SUPPORTED_SAMPLE_RATES}")
        
        # Normalize if requested
        if normalize:
            audio_data = self._normalize_audio(audio_data)
        
        # Store loaded sound
        self.loaded_sounds[file_path] = {
            'data': audio_data,
            'sample_rate': sample_rate,
            'duration': len(audio_data) / sample_rate
        }
        
        return audio_data, sample_rate
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio data to [-1, 1] range."""
        if np.max(np.abs(audio_data)) > 0:
            return audio_data / np.max(np.abs(audio_data))
        return audio_data
    
    def validate_wav_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Validate a WAV file for use as a canned sound source.
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        if not os.path.exists(file_path):
            return False, ["File does not exist"]
        
        try:
            audio_data, sample_rate = sf.read(file_path)
        except Exception as e:
            return False, [f"Error reading file: {str(e)}"]
        
        # Check channels
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            warnings.append("File is not mono - only first channel will be used")
        
        # Check sample rate
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            warnings.append(f"Sample rate {sample_rate}Hz is not in recommended list")
        
        # Check duration
        duration = len(audio_data) / sample_rate
        if duration > 10.0:  # 10 seconds max
            warnings.append("Sound duration exceeds 10 seconds - consider using shorter sounds")
        
        # Check for clipping
        if np.max(np.abs(audio_data)) >= 1.0:
            warnings.append("Audio may be clipping (samples at full scale)")
        
        return len(warnings) == 0, warnings
    
    def get_sound_info(self, file_path: str) -> Optional[dict]:
        """Get information about a loaded sound."""
        if file_path in self.loaded_sounds:
            return self.loaded_sounds[file_path].copy()
        return None
    
    def resample_audio(self, audio_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """
        Resample audio data to target sample rate.
        
        Args:
            audio_data: Original audio data
            original_rate: Original sample rate
            target_rate: Target sample rate
            
        Returns:
            Resampled audio data
        """
        from scipy import signal
        
        if original_rate == target_rate:
            return audio_data
        
        # Calculate resampling ratio
        ratio = target_rate / original_rate
        resampled_data = signal.resample(audio_data, int(len(audio_data) * ratio))
        
        return resampled_data
    
    def preprocess_sound(self, file_path: str, target_sample_rate: int = 48000) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess a sound file for simulation use.
        
        Args:
            file_path: Path to the WAV file
            target_sample_rate: Desired sample rate for simulation
            
        Returns:
            Tuple of (processed_audio_data, actual_sample_rate)
        """
        # Load and validate
        audio_data, original_rate = self.load_wav_file(file_path)
        
        # Resample if necessary
        if original_rate != target_sample_rate:
            audio_data = self.resample_audio(audio_data, original_rate, target_sample_rate)
            sample_rate = target_sample_rate
        else:
            sample_rate = original_rate
        
        return audio_data, sample_rate
    
    def batch_load_sounds(self, file_paths: List[str]) -> Dict[str, dict]:
        """
        Load multiple sound files in batch.
        
        Returns:
            Dictionary mapping file paths to sound data
        """
        results = {}
        
        for file_path in file_paths:
            try:
                audio_data, sample_rate = self.load_wav_file(file_path)
                results[file_path] = {
                    'data': audio_data,
                    'sample_rate': sample_rate,
                    'duration': len(audio_data) / sample_rate,
                    'loaded_successfully': True
                }
            except Exception as e:
                results[file_path] = {
                    'error': str(e),
                    'loaded_successfully': False
                }
        
        return results
