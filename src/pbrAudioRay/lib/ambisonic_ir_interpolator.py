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
import soundfile as sf
from scipy.signal import convolve
from scipy.interpolate import interp1d
from typing import List, Tuple, Any
from dataclasses import dataclass
import warnings

from pbrAudioRay.core.entity_manager import EntityManager
from pbrAudioRay.lib.functions import _mono_to_bands

@dataclass
class AmbisonicIRInterpolator:
    """
    Interpolates ambisonic impulse responses for smooth audio rendering.
    
    Parameters:
    -----------
    ir_sequence : np.ndarray
        Shape: (n_frames, n_channels, ir_length)
        Sequence of ambisonic IRs
    fps : float
        Frames per second of the original animation
    """
    entity_manager: EntityManager
    combo: Tuple[int, int]

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.sample_rate = int(config.system.sample_rate)
        frequency_bands = self.entity_manager.get('frequency_bands')
        n_bands = len(frequency_bands.get_bands())

        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        self.sfps = ( fps / fps_base ) * subframes # subframes per seconds

        source_idx, output_idx = combo
        for out_config in config.outputs:
            if out_config.idx == output_idx:
                ambisonic_order = out_config.order

        for src_config in config.sources:
            if src_config.idx == source_idx:
                self.audio_file = src_config.audio_file

        ir_path = f"{config.system.cache_path}/impulse_responses"
        items = os.listdir(ir_path)
        interpolators = []
        ir_sequence = []
        for bands_idx in range(n_bands):
            items = [x for x in items if x.startswith(f"ambiIR_{ambisonic_order}") and x.endswith(f"_{source_idx}_{output_idx}_{bands_idx:05}.wav")]
            filenames = sorted(items, key=lambda x: int(''.join(filter(str.isdigit, x))))
            for filename in filenames:
                ir_data, sr = sf.read(filename)
                ir_sequence += [ir_data]

                ir_sequence = np.array(ir_sequence)
                # Pre-compute interpolation functions for each channel and sample
                self.interpolators += [self._build_bands_interpolators(ir_sequence)] # per bands_idx interpolator for frame_idx
        
        # Initialize output buffer
        n_channels = (ambisonic_order + 1) ** 2 
        audio_data, sr = sf.read(self.audio_file)
        output_length = len(audio_data) + self.max_ir_length - 1
        self.output = np.zeros((n_channels, output_length))

    def _build_bands_interpolators(self, ir_sequence):
        """Build interpolation functions for each channel and sample point."""
        # Time points for original IRs (in seconds) 
        n_frames, n_channels, ir_length = ir_sequence.shape
        self.max_ir_length = max(self.max_ir_length, ir_length)
        self.frame_times = np.arange(n_frames) / self.sfps
        
        # Create interpolators for each channel and each IR sample
        interpolators = []
        for ch in range(n_channels):
            channel_irs = ir_sequence[:, ch, :]  # (n_frames, ir_length)
            # Interpolate each sample point across time
            interp_func = interp1d(
                self.frame_times,
                channel_irs.T,  # (ir_length, n_frames)
                axis=1,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            interpolators.append(interp_func)
        return interpolators
    
    def get_ir_sequence_at_times(self, time_seconds, bands_idx):
        """
        Get interpolated IR at any time point.
        
        Parameters:
        -----------
        time_seconds : float or array-like
            Time point(s) to interpolate at
        
        Returns:
        --------
        np.ndarray : Interpolated IR(s) of shape (n_channels, ir_length) or 
                    (n_times, n_channels, ir_length)
        """
        time_seconds = np.atleast_1d(time_seconds)
        
        # Clamp time to valid range
        time_seconds = np.clip(time_seconds, self.frame_times[0], self.frame_times[-1])
        
        # Get interpolated IRs for each channel
        interpolated_irs = []
        for ch in range(self.n_channels):
            ch_ir = self.interpolators[bands_idx][ch](time_seconds)  # (ir_length, n_times)
            interpolated_irs.append(ch_ir.T)  # (n_times, ir_length)
        
        result = np.stack(interpolated_irs, axis=1)  # (n_times, n_channels, ir_length)
        
        # Remove singleton dimension if single time point
        if result.shape[0] == 1:
            return result[0]
        return result
    
    def smooth_convolve(self, hop_size=None):
        """
        Smoothly convolve audio with interpolated ambisonic IRs.
        
        Parameters:
        -----------
        audio : np.ndarray
            Mono audio signal to convolve (n_samples,)
        hop_size : int, optional
            Number of audio samples between IR updates. 
            Default: one IR per audio frame (sample_rate / fps)
        
        Returns:
        --------
        np.ndarray : Convolved audio of shape (n_channels, n_output_samples)
        """
        config = self.entity_manager.get('config')
        frequency_bands = self.entity_manager.get('frequency_bands')
        n_bands = len(frequency_bands.get_bands())

        multi_bands_audio = _mono_to_bands(self.audio_file, self.sample_rate, frequency_bands)

        for bands_idx in range(n_bands):
            audio = multi_bands_audio[bands_idx]
            audio_duration = len(audio) / self.sample_rate
        
            # Calculate hop size (samples between IR updates)
            if hop_size is None:
                hop_size = int(sample_rate / self.sfps)
        
            # Calculate number of IR updates needed
            n_updates = int(np.ceil(audio_duration * self.sfps)) + 1
        
            # Time points for IR updates
            update_times = np.linspace(0, audio_duration, n_updates)
        
            # Get interpolated IRs at update times
            interpolated_irs = self.get_ir_sequence_at_times(update_times, bands_idx)
        
            # Perform overlap-add convolution
            for i, (time, ir) in enumerate(zip(update_times, interpolated_irs)):
                # Calculate start sample for this IR
                start_sample = int(time * sample_rate)
            
                # Ensure we don't exceed audio bounds
                if start_sample >= len(audio):
                    break
            
                # Extract audio segment for this IR
                end_sample = min(start_sample + hop_size, len(audio))
                audio_segment = audio[start_sample:end_sample]
            
                # Convolve each channel
                for ch in range(self.n_channels):
                    conv_result = convolve(audio_segment, ir[ch], mode='full')
                
                    # Add to output (with overlap)
                    seg_end = min(start_sample + len(conv_result), output_length)
                    seg_len = seg_end - start_sample
                    self.output[ch, start_sample:seg_end] += conv_result[:seg_len]
