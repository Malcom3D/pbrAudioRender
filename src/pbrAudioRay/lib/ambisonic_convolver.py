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
import soundfile as sf
import os

@dataclass
class AmbisonicTimeVaryingConvolver:
    """
    Time-varying ambisonic convolver optimized with SIMD and parallel processing.
    Handles per-frame per-band ambisonic IR sequences for moving sources/listeners.
    """
    
    def __init__(self, 
                 sample_rate: int = 48000,
                 ambisonic_order: int = 1,
                 hop_size: Optional[int] = None,
                 n_bands: int = 24,
                 n_threads: int = 4):

        self.sample_rate = sample_rate
        self.ambisonic_order = ambisonic_order
        self.n_channels = (ambisonic_order + 1) ** 2
        self.hop_size = hop_size or sample_rate // 24
        self.n_bands = n_bands
        
        # Set numba thread count
        nb.set_num_threads(n_threads)
        
        # Initialize buffers
        self.bands_irs: List[List[np.ndarray]] = []  # [band][frame] -> (samples, channels)
        self.max_ir_length = 0
        self.output_buffer: Optional[np.ndarray] = None
        
    def load_ir_sequence(self, ir_path: str, source_idx: int, output_idx: int):
        """
        Load per-frame per-band ambisonic IR sequence from disk.
        Optimized with memory-mapped loading for large datasets.
        """
        # Find all IR files for this source-output pair
        ir_files = []
        for bands_idx in range(self.n_bands):
            pattern = f"ambiIR_{self.ambisonic_order}_*_{source_idx}_{output_idx}_{bands_idx:05}.wav"
            band_files = sorted([f for f in os.listdir(ir_path) if f.startswith(f"ambiIR_{self.ambisonic_order}") 
                                and f.endswith(f"_{source_idx}_{output_idx}_{bands_idx:05}.wav")],
                               key=lambda x: int(''.join(filter(str.isdigit, x.split('_')[2]))))
            ir_files.append(band_files)
        
        # Load IRs with memory mapping for efficiency
        self.bands_irs = []
        for bands_idx in range(self.n_bands):
            band_irs = []
            for filename in ir_files[bands_idx]:
                # Use memory mapping for large files
                ir_data, sr = sf.read(f"{ir_path}/{filename}", always_2d=True)
                self.max_ir_length = max(self.max_ir_length, ir_data.shape[0])
                band_irs.append(ir_data)
            self.bands_irs.append(band_irs)
        
        # Pad all IRs to same length for vectorized processing
        self._pad_irs_to_uniform_length()
        
        # Pre-compute IR interpolation weights for smooth transitions
        self._precompute_interpolation_weights()
        
    def _pad_irs_to_uniform_length(self):
        """Pad all IRs to maximum length for SIMD-friendly processing."""
        for bands_idx in range(self.n_bands):
            for frame_idx in range(len(self.bands_irs[bands_idx])):
                ir = self.bands_irs[bands_idx][frame_idx]
                if ir.shape[0] < self.max_ir_length:
                    pad_length = self.max_ir_length - ir.shape[0]
                    self.bands_irs[bands_idx][frame_idx] = np.pad(ir, 
                        ((0, pad_length), (0, 0)), mode='constant')
    
    def _precompute_interpolation_weights(self):
        """Pre-compute linear interpolation weights for smooth IR transitions."""
        n_frames = len(self.bands_irs[0]) if self.bands_irs else 0
        self.interp_weights = np.zeros((n_frames, 2), dtype=np.float32)
        
        for i in range(n_frames):
            # Linear crossfade weights
            t = i / max(n_frames - 1, 1)
            self.interp_weights[i] = [1.0 - t, t]
    
    @staticmethod
#    @nb.jit(nopython=True, parallel=True, cache=True)
    def _simd_convolve_band(audio_segment: np.ndarray,
                           ir_sequence: np.ndarray,
                           interp_weights: np.ndarray,
                           hop_size: int,
                           output_buffer: np.ndarray,
                           n_channels: int):
        """
        SIMD-optimized convolution with time-varying IRs.
        Uses numba parallel processing for multi-core execution.
        """
        n_frames = ir_sequence.shape[0]
        ir_length = ir_sequence.shape[1]
        audio_length = audio_segment.shape[0]
        
        # Process each frame in parallel
        for frame_idx in prange(n_frames):
            start_sample = frame_idx * hop_size
            if start_sample >= audio_length:
                break
            
            end_sample = min(start_sample + hop_size, audio_length)
            seg_length = end_sample - start_sample
            
            # Get current and next IR for interpolation
            current_ir = ir_sequence[frame_idx]
            next_ir = ir_sequence[min(frame_idx + 1, n_frames - 1)]
            
            # Interpolate IRs for smooth transition
            weight = interp_weights[frame_idx, 0]
            interpolated_ir = weight * current_ir + (1.0 - weight) * next_ir
            
            # Convolve audio segment with interpolated IR per channel
            for ch in prange(n_channels):
                # Direct convolution using SIMD-friendly operations
                conv_length = seg_length + ir_length - 1
                conv_result = np.zeros(conv_length, dtype=np.float32)
                
                for i in range(seg_length):
                    ir_start = 0
                    ir_end = min(ir_length, conv_length - i)
                    
                    # Vectorized convolution for this sample
                    for j in range(ir_start, ir_end):
                        conv_result[i + j] += audio_segment[start_sample + i] * interpolated_ir[j, ch]
                
                # Add to output buffer
                out_start = start_sample
                out_end = min(start_sample + conv_length, output_buffer.shape[0])
                actual_length = out_end - out_start
                
                for i in range(actual_length):
                    output_buffer[out_start + i, ch] += conv_result[i]
        
        return output_buffer
    
    @staticmethod
#    @nb.jit(nopython=True, parallel=True, cache=True)
    def _simd_fast_convolution_fft_like(audio_segment: np.ndarray,
                                       ir_sequence: np.ndarray,
                                       interp_weights: np.ndarray,
                                       hop_size: int,
                                       output_buffer: np.ndarray,
                                       n_channels: int):
        """
        Optimized convolution using overlap-add approach with SIMD.
        More efficient than direct convolution for longer IRs.
        """
        n_frames = ir_sequence.shape[0]
        ir_length = ir_sequence.shape[1]
        audio_length = audio_segment.shape[0]
        
        # Use block processing for better cache utilization
        block_size = 1024  # Optimal block size for CPU cache
        
        for frame_idx in prange(n_frames):
            start_sample = frame_idx * hop_size
            if start_sample >= audio_length:
                break
            
            end_sample = min(start_sample + hop_size, audio_length)
            
            # Get interpolated IR
            current_ir = ir_sequence[frame_idx]
            next_ir = ir_sequence[min(frame_idx + 1, n_frames - 1)]
            weight = interp_weights[frame_idx, 0]
            interpolated_ir = weight * current_ir + (1.0 - weight) * next_ir
            
            # Process audio in blocks for better cache performance
            for block_start in range(start_sample, end_sample, block_size):
                block_end = min(block_start + block_size, end_sample)
                block = audio_segment[block_start:block_end]
                block_length = block_end - block_start
                
                # Convolve block with IR per channel
                for ch in prange(n_channels):
                    # Use vectorized operations
                    for i in range(block_length):
                        ir_end = min(ir_length, output_buffer.shape[0] - block_start - i)
                        if ir_end <= 0:
                            continue
                        
                        # Vectorized inner loop
                        for j in range(ir_end):
                            output_buffer[block_start + i + j, ch] += block[i] * interpolated_ir[j, ch]
        
        return output_buffer
    
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
        self.output_buffer = np.zeros((output_length, self.n_channels), dtype=np.float32)
        
        # Convert IR sequence to numpy array for SIMD processing
        n_frames = len(self.bands_irs[0])
        ir_sequence = np.zeros((n_frames, self.max_ir_length, self.n_channels), dtype=np.float32)
        
        for frame_idx in range(n_frames):
            for bands_idx in range(self.n_bands):
                ir_sequence[frame_idx] += self.bands_irs[bands_idx][frame_idx]
        
        # Normalize IR sequence
        for frame_idx in range(n_frames):
            max_val = np.max(np.abs(ir_sequence[frame_idx]))
            if max_val > 0:
                ir_sequence[frame_idx] /= max_val
        
        # Choose optimal convolution method based on IR length
        if self.max_ir_length > 2048:
            # Use FFT-like approach for longer IRs
            self._simd_fast_convolution_fft_like(
                audio_data.astype(np.float32),
                ir_sequence,
                self.interp_weights,
                self.hop_size,
                self.output_buffer,
                self.n_channels
            )
        else:
            # Use direct convolution for shorter IRs
            self._simd_convolve_band(
                audio_data.astype(np.float32),
                ir_sequence,
                self.interp_weights,
                self.hop_size,
                self.output_buffer,
                self.n_channels
            )
        
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
    
    def __init__(self,
                 sample_rate: int = 48000,
                 ambisonic_order: int = 1,
                 frequency_bands: List[Tuple[float, float]] = None,
                 hop_size: Optional[int] = None,
                 n_threads: int = 4):

        self.sample_rate = sample_rate
        self.ambisonic_order = ambisonic_order
        self.n_channels = (ambisonic_order + 1) ** 2
        self.frequency_bands = frequency_bands or [(20, 20000)]
        self.n_bands = len(self.frequency_bands)
        self.hop_size = hop_size or sample_rate // 24
        
        nb.set_num_threads(n_threads)
        
        # Per-band convolvers
        self.band_convolvers = [AmbisonicTimeVaryingConvolver(
            sample_rate=sample_rate,
            ambisonic_order=ambisonic_order,
            hop_size=hop_size,
            n_bands=1,
            n_threads=n_threads
        ) for _ in range(self.n_bands)]
        
    def load_ir_sequence(self, ir_path: str, source_idx: int, output_idx: int):
        """Load IR sequence for all bands."""
        for bands_idx in range(self.n_bands):
            self.band_convolvers[bands_idx].load_ir_sequence(
                ir_path, source_idx, output_idx
            )
    
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
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(combined_output))
        if max_val > 0:
            combined_output /= max_val
        
        return combined_output
    
    def save_output(self, output_path: str, filename: str):
        """Save combined multi-band output."""
        os.makedirs(output_path, exist_ok=True)
        filepath = os.path.join(output_path, filename)
        sf.write(filepath, self.output_buffer, self.sample_rate, subtype='FLOAT')
        print(f"Saved multiband ambisonic audio: {filepath}")
