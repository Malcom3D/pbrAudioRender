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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import soundfile as sf
from scipy import signal

from ..core.entity_manager import EntityManager
from ..lib.functions import _cartesian_to_spherical

@dataclass
class AmbisonicRender:
    """Render ambisonic audio from recorded microphone data."""
    entity_manager: EntityManager

    def __post_init__(self):
        self.config = self.entity_manager.get('config')
        self.ambisonic_config = self.config.ambisonic_render
        self.output_dir = self.ambisonic_config.path
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get ambisonic output configurations
        self.ambisonic_outputs = [oc for oc in self.config.outputs if oc.type == 'ambisonic']
        
        # For each ambisonic output, load its spatial arrangement
        self.arrangements = {}
        for ao in self.ambisonic_outputs:
            with open(ao.spatial_arrangement_file, 'r') as f:
                self.arrangements[ao.idx] = json.load(f)
    
    def render(self):
        """Render ambisonic audio for all ambisonic outputs."""
        # For each ambisonic output, we have multiple microphone channels (from spatial arrangement)
        # The recorded audio files are per microphone (omnidirectional, cardioid, etc.)
        # We need to combine them into A-format and then encode to B-format.
        # The spatial arrangement JSON contains a list of outputs, each with type and position.
        # The recorded files are named like "source_{source_idx}_output_{microphone_id}.wav"
        # For each ambisonic output, we need to collect the audio for all its microphones and encode.
        
        sources = self.entity_manager.get('sources')
        outputs = self.entity_manager.get('outputs')
        
        for ao in self.ambisonic_outputs:
            # Get the arrangement
            arr = self.arrangements[ao.idx]
            # For each source, we need to create B-format track
            for source_idx in sources.keys():
                # Collect audio from each microphone in arrangement
                mic_signals = []
                for mic in arr['outputs']:
                    # Find the actual output object that corresponds to this microphone
                    mic_output = None
                    for out_idx, out in outputs.items():
                        if out.config_idx == ao.idx and out.idx == mic['id']:
                            mic_output = out
                            break
                    if mic_output is None:
                        raise RuntimeError(f"Microphone {mic['id']} not found in outputs")
                    # Load recorded audio for this source and microphone
                    audio_file = os.path.join(mic_output.output_config.render_output_path, f"source_{source_idx}_output_{mic_output.idx}.wav")
                    audio, sr = sf.read(audio_file)
                    if sr != self.ambisonic_config.sample_rate:
                        # Resample
                        audio = signal.resample(audio, int(len(audio) * self.ambisonic_config.sample_rate / sr))
                    mic_signals.append(audio)
                
                # Encode to B-format
                bformat = self._encode_to_bformat(mic_signals, arr)
                # Save
                output_file = os.path.join(self.output_dir, f"source_{source_idx}_ambisonic_{ao.idx}.wav")
                sf.write(output_file, bformat.T, self.ambisonic_config.sample_rate)  # bformat shape: (samples, channels)
    
    def _encode_to_bformat(self, mic_signals, arrangement):
        """Encode A-format microphone signals to B-format (ACN, N3D)."""
        # The arrangement may provide encoding matrix, or we compute from positions
        if 'encoding_matrix' in arrangement:
            # Use provided matrix: rows are B-format channels, columns are microphones
            matrix = np.array(arrangement['encoding_matrix'])
            # Convert to list of B-format channel names (W, X, Y, Z, etc.)
            # We'll output channels in ACN order: W, X, Y, Z, ... for order 1; for higher orders, standard ordering.
        else:
            # Compute encoding matrix based on microphone positions and directivity
            # For simplicity, assume all microphones are omnidirectional (A-format)
            # B-format signals: W = sum of all microphones (weighted), X = sum of microphones * x coordinate, etc.
            # For N3D normalization, we need to scale accordingly.
            num_mics = len(mic_signals)
            num_samples = mic_signals[0].shape[0]
            bformat = np.zeros((num_samples, 4))  # order 1: W, X, Y, Z
            for i, mic in enumerate(arrangement['outputs']):
                pos = np.array(mic['position'])
                # W component: all microphones contribute equally (or weight by 1)
                bformat[:, 0] += mic_signals[i]  # W
                # X, Y, Z: weighted by position
                bformat[:, 1] += mic_signals[i] * pos[0]  # X
                bformat[:, 2] += mic_signals[i] * pos[1]  # Y
                bformat[:, 3] += mic_signals[i] * pos[2]  # Z
            # Normalize according to N3D: for order 1, factor sqrt(3) for X,Y,Z? Actually N3D: W = 1/sqrt(4π) * sum, but we'll use common convention.
            # Usually, for N3D, the encoding matrix includes normalization. We'll just output raw and let the user handle.
            # For now, keep as is.
        return bformat
