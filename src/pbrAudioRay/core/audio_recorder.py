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
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from dask import delayed, compute

from ..core.entity_manager import EntityManager
from ..outputs.omnidirectional_output import OmnidirectionalOutput
from ..outputs.cardioid_output import CardioidOutput
from ..outputs.hypercardioid_output import HypercardioidOutput
from ..outputs.figure8_output import Figure8Output
from ..outputs.ambisonic_output import AmbisonicOutput

@dataclass
class AudioRecorder:
    entity_manager: EntityManager

    def __post_init__(self):
        self.config = self.entity_manager.get('config')
        for output_config in self.config.outputs:
            output_dir = output_config.render_output_path
            os.makedirs(output_dir, exist_ok=True)
        self.audio_recorder_config = self.config.audio_recorder
        self.sample_rate = self.config.system.sample_rate
    
    def compute(self):
        """Process audio for all microphones and save to files."""
        # Get all sources
        sources = self.entity_manager.get('sources')
        outputs = self.entity_manager.get('outputs')
        
        tasks = []
        for source_idx, source in sources.items():
            # Load source audio
            source_config = None
            for sc in self.config.sources:
                if sc.idx == source_idx:
                    source_config = sc
                    break
            if source_config.audio_file:
                import soundfile as sf
                audio, sr = sf.read(source_config.audio_file)
                if sr != self.sample_rate:
                    # Resample
                    from scipy import signal
                    audio = signal.resample(audio, int(len(audio) * self.sample_rate / sr))
            else:
                # Generate test tone? For now, zeros
                audio = np.zeros(int(self.sample_rate * 5))  # 5 seconds silence
            
            for output_idx, output in outputs.items():
                output_config = output.output_config  # each output has its config
                output_file = os.path.join(output_config.render_output_path, f"source_{source_idx}_output_{output_idx}.wav")
                # Convolve
                tasks.append(delayed(output.record_audio)(audio, source_idx, output_file))
        
        compute(*tasks)
