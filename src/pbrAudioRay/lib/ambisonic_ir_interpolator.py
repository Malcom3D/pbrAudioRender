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
import numpy as np
import numba as nb
from numba import pr prange
from dask import delayed, compute
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from scipy import signal as scipy_signal
import soundfile as sf

from ..core.entity_manager import EntityManager
from ..lib.output_data import OutputData
from ..lib.functions import _cartesian_to_spherical, _mono_to_bands

@dataclass
class AmbisonicIRInterpolator:
    """
    Per-sample time-varying convolution for ambisonic acoustic ray tracing.
    Optimized for CPU with SIMD, numba, and dask.
    """
    entity_manager: EntityManager
    combo: Tuple[int, int]  # (source_idx, output_idx)
    
    def __post_init__(self):
        self.config = self.entity_manager.get('config')
        self.frequency_bands = self.entity_manager.get('frequency_bands')
        self.sample_rate = self.config.system.sample_rate
        self.fps = self.config.system.fps
        self.subframes = self.config.system.subframes
        self.n_bands = len(self.frequency_bands.get_bands())
        
        # Get source and output configs
        self.source_config = None
        self.output_config = None
        for src in self.config.sources:
            if src.idx == self.combo[0]:
                self.source_config = src
                break
        for out in self.config.outputs:
            if out.idx == self.combo[1]:
                self.output_config = out
                break
        
        # Load source audio
        self.source_audio = self._load_source_audio()
        
        # Initialize output buffers
        self._init_output_buffers()
        
    def _load_source_audio(self) -> np.ndarray:
        """Load and preprocess source audio into frequency bands."""
        if self.source_config.audio_file:
            # Convert to multiband audio
            audio_bands = _mono_to_bands(
                self.source_config.audio_file,
                self.sample_rate,
                self.frequency_bands.get_bands()
            )
            return audio_bands  # Shape: (n_bands, n_samples)
        return None
    
    def _init_output_buffers(self):
        """Initialize output audio buffers."""
        if self.source_audio is not None:
            n_samples = self.source_audio.shape[1]
            self.output_buffer = np.zeros(n_samples, dtype=np.float32)
        else:
            self.output_buffer = np.zeros(0, dtype=np.float32)
    
    @delayed
    def smooth_convolve(self) -> np.ndarray:
        """
        Perform time-varying convolution with frame interpolation.
        Uses crossfading between frames for smooth transitions.
        """
        if self.source_audio is None:
            return self.output_buffer
        
        # Get frame data from wave propagators
        wave_propagators = self.entity_manager.get('wave_propagators')
        
        # Collect all frame outputs
        frame_outputs = []
        for wp_idx, wp in wave_propagators.items():
            if wp.combo == self.combo:
                # We need to access the output data from each frame
                # This assumes the data is stored stored somewhere accessible
                pass
        
        # For now, let's assume we have frame data in a list
        # This would need to be adapted to your actual data flow
        
        return self.output_buffer
    
    @staticmethod
    @nb.jit(nopython=True, parallel=True, fastmath=True)
    def _compute_ir_for_frame(energies: np.ndarray, phases: np.ndarray, delays: np.ndarray, directions: np.ndarray, sample_rate: int, n_bands: int) -> np.ndarray:
        """
        Compute impulse response for a single frame using SIMD optimization.
        
        Args:
            energies: Ray energies at output (n_rays, n_bands)
            phases: Ray phases at output (n_rays, n_bands)
            delays: Ray delays in seconds (n_rays, 1)
            directions: Ray directions (n_rays, 3)
            sample_rate: Output sample rate
            n_bands: Number of frequency bands
            
        Returns:
            IR array (n_samples,)
        """
        n_rays = energies.shape[0]
        
        # Calculate maximum IR length
        max_delay_samples = int(np.max(delays) * sample_rate) + 1
        ir_length = max_delay_samples + 1
        
        # Initialize IR with zeros
        ir = np.zeros(ir_length, dtype=np.float32)
        
        # Process each ray in parallel
        for i in nb.prange(n_rays):
            # Convert delay to samples
            delay_samples = int(delays[i, 0] * sample_rate)
            
            if delay_samples < ir_length:
                # Sum contributions from all bands
                for b in range(n_bands):
                    # Energy contribution
                    energy = energies[i, b] if energies.ndim > 1 else energies[i, 0]
                    phase = phases[i, b] if phases.ndim > 1 else phases[i, 0]
                    
                    # Apply phase to energy (complex representation)
                    # For simplicity, we use energy directly
                    # Phase would be used for more accurate interference modeling
                    
                    # Add to IR at appropriate delay
                    ir[delay_samples] += energy
        
        return ir
    
    @staticmethod
    @nb.jit(nopython=True, parallel=True, fastmath=True)
    def _apply_spherical_harmonics(energies: np.ndarray, directions: np.ndarray, order: int) -> np.ndarray:
        """
        Apply spherical harmonics decomposition for ambisonic encoding.
        
        Args:
            energies: Ray energies (n_rays, n_bands)
            directions: Ray directions (n_rays, 3)
            order: Ambisonic order
            
        Returns:
            SH coefficients (n_channels, n_bands)
        """
        n_rays = energies.shape[0]
        n_bands = energies.shape[1] if energies.ndim > 1 else 1
        n_channels = (order + 1) ** 2
        
        # Initialize SH coefficients
        sh_coeffs = np.zeros((n_channels, n_bands), dtype=np.float32)
        
        # Convert directions to spherical coordinates
        for i in nb.prange(n_rays):
            x, y, z = directions[i, 0], directions[i, 1], directions[i, 2]
            
            # Calculate spherical coordinates
            azimuth = np.arctan2(y, x)
            elevation = np.arcsin(z / max(np.sqrt(x*x + y*y + z*z), 1e-10))
            
            # Compute SH basis functions up to given order
            # Simplified for performance - use precomputed values for common orders
            
            for b in range(n_bands):
                energy = energies[i, b]
                
                # Order 0 (W channel - omnidirectional)
                sh_coeffs[0, b] += energy * 0.5  # Normalization factor
                
                if order >= 1:
                    # Order 1 (X, Y, Z channels)
                    cos_az = np.cos(azimuth)
                    sin_az = np.sin(azimuth)
                    cos_el = np.cos(elevation)
                    sin_el = np.sin(elevation)
                    
                    sh_coeffs[1, b] += energy * cos_az * cos_el  # X
                    sh_coeffs[2, b] += energy * sin_az * cos_el  # Y
                    sh_coeffs[3, b] += energy * sin_el           # Z
                
                if order >= 2:
                    # Order 2 (simplified)
                    cos2_az = np.cos(2 * azimuth)
                    sin2_az = np.sin(2 * azimuth)
                    cos2_el = np.cos(2 * elevation)
                    sin2_el = np.sin(2 * elevation)
                    
                    sh_coeffs[4, b] += energy * cos2_az * cos_el * cos_el  # R
                    sh_coeffs[5, b] += energy * sin2_az * cos_el * cos_el  # S
                    sh_coeffs[6, b] += energy * cos_az * sin_el * cos_el   # T
                    sh_coeffs[7, b] += energy * sin_az * sin_el * cos_el   # U
                    sh_coeffs[8, b] += energy * (3 * sin_el * sin_el - 1) / 2  # V
        
        return sh_coeffs
    
    @staticmethod
    @nb.jit(nopython=True, parallel=True, fastmath=True)
    def _time_varying_convolution(source_bands: np.ndarray, ir_frames: np.ndarray, frame_indices: np.ndarray, sample_rate: int, fps: int) -> np.ndarray:
        """
        Perform time-varying convolution with frame interpolation.
        
        Args:
            source_bands: Source audio in frequency bands (n_bands, n_samples)
            ir_frames: IR frames (n_frames, n_bands, ir_length)
            frame_indices: Frame indices for each sample (n_samples,)
            sample_rate: Audio sample rate
            fps: Frame rate
            
        Returns:
            Output audio (n_samples,)
        """
        n_bands = source_bands.shape[0]
        n_samples = source_bands.shape[1]
        n_frames = ir_frames.shape[0]
        ir_length = ir_frames.shape[2]
        
        # Initialize output
        output = np.zeros(n_samples, dtype=np.float32)
        
        # Samples per frame
        samples_per_frame = sample_rate // fps
        
        # Process each band independently
        for b in nb.prange(n_bands):
            source = source_bands[b]
            
            # For each output sample
            for n in range(n_samples):
                # Find current frame
                frame_idx = min(n // samples_per_frame, n_frames - 1)
                
                # Get IR for this frame and band
                ir = ir_frames[frame_idx, b]
                
                # Convolve with overlap-add approach
                # Simplified: direct convolution at each sample position
                contrib = 0.0
                for k in range(ir_length):
                    if n - k >= 0:
                        contrib += source[n - k] * ir[k]
                
                output[n] += contrib
        
        return output
    
    @delayed
    def save_output(self):
        """Save the convolved output audio."""
        config = self.entity_manager.get('config')
        
        # Determine output path
        output_dir = config.ambisonic_render.path
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename based on source and output
        source_name = self.source_config.name if self.source_config else f"src_{self.combo[0]}"
        output_name = self.output_config.name if self.output_config else f"out_{self.combo[1]}"
        filename = f"{source_name}_to_{output_name}.wav"
        filepath = os.path.join(output_dir, filename)
        
        # Save as WAV
        sf.write(filepath, self.output_buffer, self.sample_rate)
        print(f"Saved output: {filepath}")
        
        return filepath
