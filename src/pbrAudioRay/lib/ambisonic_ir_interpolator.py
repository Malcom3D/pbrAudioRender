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
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from pbrAudioCommon import EntityManager
from pbrAudioCommon import _mono_to_bands

from .ambisonic_convolver import AmbisonicTimeVaryingConvolver, MultibandAmbisonicConvolver

@dataclass
class AmbisonicIRInterpolator:
    """
    Optimized ambisonic IR interpolator using SIMD-accelerated time-varying convolution.
    """
    entity_manager: EntityManager
    combo: Tuple[int, int]
    use_multiband: bool = False  # Toggle for single vs multiband processing
    n_threads: int = 4
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.sample_rate = int(config.system.sample_rate)
        self.frequency_bands = self.entity_manager.get('frequency_bands')
        self.n_bands = len(self.frequency_bands.get_bands())

        if config.system.bands_per_octave > 0:
            use_multiband = True
        
        fps = config.system.fps
        fps_base = config.system.fps_base
        subframes = config.system.subframes
        self.sfps = (fps / fps_base) * subframes
        
        source_idx, output_idx = self.combo
        
        # Get output configuration
        for out_config in config.outputs:
            if out_config.idx == output_idx:
                ambisonic_order = out_config.order
                self.n_channels = (ambisonic_order + 1) ** 2
                self.out_name = out_config.name
                break
        
        # Get source configuration
        for src_config in config.sources:
            if src_config.idx == source_idx:
                self.audio_file = src_config.audio_file
                self.src_name = src_config.name
                break
        
        # Initialize convolver
        if self.use_multiband:
            self.convolver = MultibandAmbisonicConvolver(sample_rate=self.sample_rate, ambisonic_order=ambisonic_order, frequency_bands=self.frequency_bands.get_bands(), hop_size=int(self.sample_rate / self.sfps), n_threads=self.n_threads)
        else:
            self.convolver = AmbisonicTimeVaryingConvolver(sample_rate=self.sample_rate, ambisonic_order=ambisonic_order, hop_size=int(self.sample_rate / self.sfps), n_threads=self.n_threads)
        # Load IR sequence
        ir_path = f"{config.system.cache_path}/impulse_responses"
        self.convolver.load_ir_sequence(ir_path, source_idx, output_idx)
        
        # Initialize output buffer
        if self.audio_file.endswith('.wav'):
            audio_data, sr = sf.read(self.audio_file)
        elif self.audio_file.endswith('.raw'):
            audio_data, sr = sf.read(self.audio_file, channels=1, samplerate=self.sample_rate, subtype='FLOAT')
        self.output_length = audio_data.shape[0] + self.convolver.max_ir_length - 1
        self.output = None
    
    def smooth_convolve(self):
        """Perform time-varying multiband convolution."""
        # Read audio file
        if self.audio_file.endswith('.wav'):
            audio_data, sr = sf.read(self.audio_file)
        elif self.audio_file.endswith('.raw'):
            audio_data, sr = sf.read(self.audio_file, channels=1, samplerate=self.sample_rate, subtype='FLOAT')
        
        # Ensure mono
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sr != self.sample_rate:
            import resampy
            audio_data = resampy.resample(audio_data, sr, self.sample_rate)
        
        # Perform convolution
        self.output = self.convolver.convolve(audio_data)
        
        return self.output
    
    def save_output(self):
        """Save convolved audio to file."""
        if self.output is None:
            raise RuntimeError("No output to save. Call smooth_convolve() first.")
        
        config = self.entity_manager.get('config')
        render_path = config.system.render_path
        os.makedirs(render_path, exist_ok=True)
        
        filename = f"{self.src_name}_{self.out_name}.wav"
        filepath = os.path.join(render_path, filename)
        
        sf.write(filepath, self.output, self.sample_rate, subtype='FLOAT')
        print(f"Saved convolved Audio: {filepath} for source {self.src_name}, output {self.out_name}")
