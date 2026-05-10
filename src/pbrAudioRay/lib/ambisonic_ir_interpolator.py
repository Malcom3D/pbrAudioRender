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
import numpy as np
import soundfile as sf
from scipy.signal import convolve
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from pbrAudioRay.core.entity_manager import EntityManager
from pbrAudioRay.lib.functions import _mono_to_bands

@dataclass
class AmbisonicIRInterpolator:
    entity_manager: EntityManager
    combo: Tuple[int, int]
    interpolators: List[List] = field(default_factory=lambda: [[]])
    max_ir_length: int = 0
    n_channels: int = 0

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.sample_rate = int(config.system.sample_rate)
        frequency_bands = self.entity_manager.get('frequency_bands')
        n_bands = len(frequency_bands.get_bands())

        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        self.sfps = ( fps / fps_base ) * subframes # subframes per seconds

        source_idx, output_idx = self.combo
        for out_config in config.outputs:
            if out_config.idx == output_idx:
                ambisonic_order = out_config.order
                self.n_channels = (ambisonic_order + 1) ** 2
                self.out_name = out_config.name

        for src_config in config.sources:
            if src_config.idx == source_idx:
                self.audio_file = src_config.audio_file
                self.src_name = src_config.name

        ir_path = f"{config.system.cache_path}/impulse_responses"
        for bands_idx in range(n_bands):
            ir_sequence = []
            items = os.listdir(ir_path)
            items = [x for x in items if x.startswith(f"ambiIR_{ambisonic_order}") and x.endswith(f"_{source_idx}_{output_idx}_{bands_idx:05}.wav")]
            filenames = sorted(items, key=lambda x: int(''.join(filter(str.isdigit, x))))
            for filename in filenames:
                ir_data, sr = sf.read(f"{ir_path}/{filename}")
                n_frames, n_channels = ir_data.shape
                self.max_ir_length = max(self.max_ir_length, n_frames)
                ir_sequence += [ir_data]

            # Pre-compute interpolation functions for each channel and sample
            self.interpolators += [self._build_interpolator(ir_sequence)]

        # Initialize output buffer
        audio_data, sr = sf.read(self.audio_file)
        output_length = len(audio_data) + self.max_ir_length - 1
        self.output = np.zeros((output_length, self.n_channels))

    def _build_interpolator(self, ir_sequence: List[np.ndarray]) -> RegularGridInterpolator:
        """
        Returns:
            Interpolated impulse response (samples x channels)
        """
        # Ensure all irs have the same length
        ir_datas = []
        for ir_data in ir_sequence:
            if not ir_data.shape[0] == self.max_ir_length:
                diff_samples = self.max_ir_length - ir_data.shape[0]
                ir_data = np.append(ir_data, np.zeros((diff_samples, ir_data.shape[1])), axis=0)
            ir_datas += [ir_data]

        # Create a 3D grid: (time_position, sample, channel)
        # Frame time positions in samples of IRs
        times = np.arange(len(ir_datas)) * self.sample_rate / self.sfps
        
        # Sample indices
        samples = np.arange(self.max_ir_length)
        
        # Channel indices
        channels = np.arange(self.n_channels)
        
        # Create the grid
        grid = (times, samples, channels)
        
        # Create values array
        values = np.stack(ir_datas, axis=0)
        
        # Create interpolator
        interpolator = RegularGridInterpolator(grid, values, method='slinear', bounds_error=False, fill_value=None)
        return interpolator

    def get_ir_sequence_at_times(self, time_samples: np.ndarray, bands_idx: int):
        # Create points to interpolate
        interpolator = self.interpolators[bands_idx]
        interpolated_irs = []
        for idx in range(time_samples.shape[0]):
            points = np.array([[time_samples[idx], s, c] for s in range(self.max_ir_length) for c in range(self.n_channels)])

            # Perform interpolation
            interpolated_ir = interpolator(points)

            # Reshape back to (samples, channels)
            interpolated_irs += interpolated_ir.reshape(self.max_ir_length, self.n_channels)
        return interpolated_irs

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
        np.ndarray : Convolved audio of shape (n_output_samples, n_channels)
        """
        config = self.entity_manager.get('config')
        frequency_bands = self.entity_manager.get('frequency_bands')
        n_bands = len(frequency_bands.get_bands())

        multi_bands_audio = _mono_to_bands(self.audio_file, self.sample_rate, frequency_bands.get_bands())

        for bands_idx in range(n_bands):
            audio = multi_bands_audio[bands_idx]
            audio_duration = len(audio)

            # Calculate hop size (samples between IR updates)
            if hop_size is None:
                hop_size = int(self.sample_rate / self.sfps)

            # Calculate number of IR updates needed
            n_updates = int(np.ceil(audio_duration / hop_size)) + 1

            # Time points for IR updates
            update_times = np.linspace(0, audio_duration, n_updates)

            # Get interpolated IRs at update times
            interpolated_irs = self.get_ir_sequence_at_times(update_times, bands_idx)

            # Perform overlap-add convolution
            for i, (start_sample, ir) in enumerate(zip(update_times, interpolated_irs)):
                # Ensure we don't exceed audio bounds
                if start_sample >= len(audio):
                    break

                # Extract audio segment for this IR
                end_sample = min(start_sample + hop_size, len(audio))
                audio_segment = audio[start_sample:end_sample]

                # Convolve each channel
                for ch in range(self.n_channels):
                    conv_result = convolve(audio_segment, ir[:,ch], mode='full')

                    # Add to output (with overlap)
                    seg_end = min(start_sample + len(conv_result), output_length)
                    seg_len = seg_end - start_sample
                    self.output[start_sample:seg_end, ch] += conv_result[seg_len:]

    def save_output(self):
        config = self.entity_manager.get('config')
        render_path = config.system.render_path
        subtype = config.system.bit_depth
        file_format = config.system.file_format.lower()
        os.makedirs(render_path, exist_ok=True)
        filename = f"{self.src_name}_{self.out_name}.{file_format}"
        sf.write(filename, self.output, self.sample_rate, subtype=subtype)

        print(f"Saved convolved Audio: {filename} for source {self.src_name}, output {self.out_name}")
