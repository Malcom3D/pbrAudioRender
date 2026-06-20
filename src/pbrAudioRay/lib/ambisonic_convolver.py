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
import numba as nb
from numba import prange
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from scipy.signal import fftconvolve
import soundfile as sf
import os

@dataclass
class AmbisonicTimeVaryingConvolver:
    """
    Time-varying ambisonic convolver optimized with SIMD and parallel processing.
    Handles per-frame single band ambisonic IR sequences for moving sources/listeners.
    """
    
    def __init__(self, sample_rate: int = 48000, ambisonic_order: int = 1, hop_size: Optional[int] = None, bands_idx: int = 0, n_threads: int = 4):

        self.sample_rate = sample_rate
        self.ambisonic_order = ambisonic_order
        self.n_channels = (ambisonic_order + 1) ** 2
        self.hop_size = hop_size or sample_rate // 24
        self.bands_idx = bands_idx
        
        # Set numba thread count
        nb.set_num_threads(n_threads)
        
        # Initialize buffers
        self.bands_irs: List[np.ndarray] = []  # [frame] -> (samples, channels)
        self.max_ir_length = 0
        self.output_buffer: Optional[np.ndarray] = None
        
    def load_ir_sequence(self, ir_path: str, source_idx: int, output_idx: int):
        """
        Load per-frame ambisonic IR sequence from disk.
        Optimized with memory-mapped loading for large datasets.
        """
        # Find all IR files for this source-output pair
        pattern = f"ambiIR_{self.ambisonic_order}_*_{source_idx}_{output_idx}_{self.bands_idx:05}.wav"
        ir_files = sorted([f for f in os.listdir(ir_path) if f.startswith(f"ambiIR_{self.ambisonic_order}") and f.endswith(f"_{source_idx}_{output_idx}_{self.bands_idx:05}.wav")], key=lambda x: int(''.join(filter(str.isdigit, x.split('_')[2]))))
        
        # Load IRs with memory mapping for efficiency
        self.bands_irs = []
        for filename in ir_files:
            # Use memory mapping for large files
            ir_data, sr = sf.read(f"{ir_path}/{filename}", always_2d=True)
            self.max_ir_length = max(self.max_ir_length, ir_data.shape[0])
        self.bands_irs.append(ir_data)
        
        # Pad all IRs to same length for vectorized processing
        self._pad_irs_to_uniform_length()
        
    def _pad_irs_to_uniform_length(self):
        """Pad all IRs to maximum length for SIMD-friendly processing."""
        for frame_idx in range(len(self.bands_irs)):
            ir = self.bands_irs[frame_idx]
            if ir.shape[0] < self.max_ir_length:
                pad_length = self.max_ir_length - ir.shape[0]
                self.bands_irs[frame_idx] = np.pad(ir, ((0, pad_length), (0, 0)), mode='constant')

    def _time_varing_convolve(self, audio_data, ir_sequence):
        n_frames = len(ir_sequence)
        ir_length = ir_sequence[0].shape[0]
        audio_length = audio_data.shape[0]
        output_length = self.output_buffer.shape[0]
        fade_in = np.linspace(0.0, 1.0, int(self.hop_size/2))
        fade_out = np.linspace(1.0, 0.0, int(self.hop_size/2))
        for frame_idx in range(n_frames):
            start_sample = int((frame_idx * self.hop_size) - (self.hop_size / 2))
            start_sample = start_sample if start_sample > 0 else 0
            if start_sample >= audio_length:
                break
            end_sample = int(min(start_sample + (3/2 * self.hop_size), audio_length))
            if frame_idx == (n_frames -1) and (end_sample + self.hop_size) < audio_length:
                end_sample = audio_length
            audio_block = audio_data[start_sample:end_sample].reshape(-1,)
            for ch in range(self.n_channels):
                ir = ir_sequence[frame_idx][:,ch]
                convolved = fftconvolve(audio_block, ir, mode='full')
                if not (end_sample == audio_length) or not (frame_idx == (n_frames -1) and (end_sample + self.hop_size) < audio_length):
                    fade_start = int(end_sample-(self.hop_size/2))
                    convolved[fade_start:end_sample] *= fade_out
                    conv_end = end_sample
                else:
                    conv_end = output_length
                if start_sample > 0:
                    fade_end = int(start_sample+(self.hop_size/2))
                    convolved[start_sample:fade_end] *= fade_in
                conv_end = int(convolved.shape[0] + start_sample)
                self.output_buffer[start_sample:conv_end, ch] += convolved.astype(np.float64)

    def convolve(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Perform time-varying ambisonic convolution.
        
        Args:
            audio_data: Input mono audio signal
            
        Returns:
            Multi-channel ambisonic audio signal
        """
        if not self.bands_irs:
            raise ValueError("No IR sequence loaded. Call load_ir_sequence() first.")
        
        audio_length = audio_data.shape[0]
        output_length = audio_length + self.max_ir_length - 1
        
        # Initialize output buffer
        self.output_buffer = np.zeros((output_length, self.n_channels), dtype=np.float64)
        
        # Convert IR sequence to numpy array for SIMD processing
        n_frames = len(self.bands_irs)
        ir_sequence = np.zeros((n_frames, self.max_ir_length, self.n_channels), dtype=np.float32)
        
        for frame_idx in range(n_frames):
            ir_sequence[frame_idx] += self.bands_irs[frame_idx]
        
        # Normalize IR sequence
        for frame_idx in range(n_frames):
            max_val = np.max(np.abs(ir_sequence[frame_idx]))
            if max_val > 0:
                ir_sequence[frame_idx] /= max_val
        
        self._time_varing_convolve(audio_data.astype(np.float32), ir_sequence)
        return self.output_buffer
    
    def save_output(self, output_path: str, filename: str):
        """Save convolved audio to file."""
        os.makedirs(output_path, exist_ok=True)
        filepath = os.path.join(output_path, filename)
        sf.write(filepath, self.output_buffer, self.sample_rate, subtype='FLOAT')
        print(f"Saved ambisonic audio: {filepath}")


# Optimized version with frequency band processing
@dataclass
class MultibandAmbisonicConvolver:
    """
    Multi-band time-varying ambisonic convolver.
    Processes each frequency band independently for better spectral accuracy.
    """
    
    def __init__(self,sample_rate: int = 48000, ambisonic_order: int = 1, frequency_bands: List[Tuple[float, float]] = None, hop_size: Optional[int] = None, n_threads: int = 4):

        self.sample_rate = sample_rate
        self.ambisonic_order = ambisonic_order
        self.n_channels = (ambisonic_order + 1) ** 2
        self.frequency_bands = frequency_bands or [(20, 20000)]
        self.n_bands = len(self.frequency_bands)
        self.hop_size = hop_size or sample_rate // 24
        
        nb.set_num_threads(n_threads)
        
        # Per-band convolvers
        self.band_convolvers = [AmbisonicTimeVaryingConvolver(sample_rate=sample_rate, ambisonic_order=ambisonic_order, hop_size=hop_size, bands_idx=bands_idx, n_threads=n_threads) for bands_idx in range(self.n_bands)]
        
    def load_ir_sequence(self, ir_path: str, source_idx: int, output_idx: int):
        """Load IR sequence for all bands."""
        for bands_idx in range(self.n_bands):
            self.band_convolvers[bands_idx].load_ir_sequence(ir_path, source_idx, output_idx)
    
    def convolve(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Perform multi-band time-varying ambisonic convolution.
        Processes each band independently and sums the results results.
        """
        output_length = audio_data.shape[0] + self.band_convolvers[0].max_ir_length - 1
        combined_output = np.zeros((output_length, self.n_channels), dtype=np.float32)
        
        # Process each band
        for bands_idx in range(self.n_bands):
            band_output = self.band_convolvers[bands_idx].convolve(audio_data)
            combined_output += band_output
        
#        # Normalize to prevent clipping
#        max_val = np.max(np.abs(combined_output))
#        if max_val > 0:
#            combined_output /= max_val
        
        return combined_output
    
    def save_output(self, output_path: str, filename: str):
        """Save combined multi-band output."""
        os.makedirs(output_path, exist_ok=True)
        filepath = os.path.join(output_path, filename)
        sf.write(filepath, self.output_buffer, self.sample_rate, subtype='FLOAT')
        print(f"Saved multiband ambisonic audio: {filepath}")
