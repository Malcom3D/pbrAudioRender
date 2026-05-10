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
fromfrom numba import prange
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class TimeVaryingConvolver:
    """
    Efficient time-varying convolution for frame-based IR data.
    Uses overlap-add with frame interpolation for smooth transitions.
    """
    sample_rate: int = 48000
    fps: int = 24
    fft_size: int = 1024
    hop_size: int = 512
    
    def __post_init__(self):
        self.samples_per_frame = self.sample_rate // self.fps
        
        # Precompute FFT plan
        self._init_fft()
    
    def _init_fft(self):
        """Initialize FFT buffers and window."""
        self.fft_size = 2 ** int(np.ceil(np.log2(self.fft_size)))
        self.hop_size = min(self.hop_size, self.fft_size // 4)
        
        # Hann window for overlap-add
        self.window = np.hanning(self.fft_size)
    
    @staticmethod
    @nb.jit(nopython=True, fastmath=True, parallel=True)
    def _overlap_add_convolution(source: np.ndarray, ir_frames: np.ndarray, frame_indices: np.ndarray, fft_size: int, hop_size: int) -> np.ndarray:
        """
        Perform overlap-add convolution with time-varying IR IR.
        
        Args:
            source source: Input audio (n_samples,)
            ir_frames: IR frames (n_frames, ir_length)
            frame_indices: Frame index for each sample (n_samples,)
            fft_size: FFT size for convolution
            hop_size: Hop size for overlap-add
            
        Returns:
            Convolved output (n_samples,)
        """
        n_samples = len(source)
        n_frames = ir_frames.shape[0]
        ir_length = ir_frames.shape[1]
        
        # Initialize output
        output = np.zeros(n_samples + ir_length - 1, dtype=np.float32)
        
        # Process in blocks
        n_blocks = (n_samples + hop_size - 1) // hop_size
        
        for block_idx in nb.prange(n_blocks):
            start = block_idx * hop_size
            end = min(start + fft_size, n_samples)
            
            if start >= n_samples:
                continue
            
            # Get current frame
            frame_idx = min(start // (n_samples // n_frames), n_frames - 1)
            ir = ir_frames[frame_idx]
            
            # Get input block
            block = source[start:end]
            block = np.pad(block, (0, fft_size - len(block)))
            
            # FFT convolution
            fft_block = np.fft.rfft(block)
            fft_ir = np.fft.rfft(ir, n=fft_size)
            
            # Multiply in frequency domain
            fft_result = fft_block * fft_ir
            
            # Inverse FFT
            result = np.fft.irfft(fft_result)[:fft_size]
            
            # Overlap-add with window
            for i in range(fft_size):
                if start + i < len(output):
                    output[start + i] += result[i]
        
        return output[:n_samples]
    
    @staticmethod
    @nb.jit(nopython=True, fastmath=True, parallel=True)
    def _crossfade_frames(ir_frames: np.ndarray, crossfade_length: int) -> np.ndarray:
        """
        Smoothly interpolate between consecutive IR frames.
        
        Args:
            ir_frames: IR frames (n_frames, ir_length)
            crossfade_length: Number of samples for crossfade
            
        Returns:
            Smoothed IR frames (n_frames, ir_length)
        """
        n_frames = ir_frames.shape[0]
        ir_length = ir_frames.shape[1]
        
        # Create crossfade window
        fade_in = np.linspace(0, 1, crossfade_length)
        fade_out = np.linspace(1, 0, crossfade_length)
        
        # Smooth frames
        smoothed = ir_frames.copy()
        
        for f in nb.prange(n_frames - 1):
            for i in range(min(crossfade_length, ir_length)):
                weight = fade_in[i]
                smoothed[f, i] = ir_frames[f, i] * (1 - weight) + ir_frames[f+1, i] * weight
        
        return smoothed
    
    def convolve(self, source: np.ndarray, ir_frames: np.ndarray, frame_times: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform time-varying convolution.
        
        Args:
            source: Input audio (n_samples,)
            ir_frames: IR frames (n_frames, ir_length)
            frame_times: Frame times in seconds (optional)
            
        Returns:
            Convolved output (n_samples,)
        """
        n_samples = len(source)
        n_frames = ir_frames.shape[0]
        
        # Calculate frame indices for each sample
        if frame_times is not None:
            frame_indices = np.searchsorted(frame_times * self.sample_rate, np.arange(n_samples))
            frame_indices = np.clip(frame_indices, 0, n_frames - 1)
        else:
            frame_indices = np.arange(n_samples) // self.samples_per_frame
            frame_indices = np.clip(frame_indices, 0, n_frames - 1)
        
        # Apply crossfade for smooth transitions
        crossfade_length = min(self.samples_per_frame // 4, 1024)
        smoothed_ir = self._crossfade_frames(ir_frames, crossfade_length)
        
        # Perform convolution
        output = self._overlap_add_convolution(source, smoothed_ir, frame_indices, self.fft_size, self.hop_size)
        
        return output

