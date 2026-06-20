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
from pbrAudioCommon import np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from pathlib import Path

from dask import delayed, compute

from ..core.entity_manager import EntityManager
from ..lib.functions import _append_to_npz

@dataclass
class AudioRecorder:
    entity_manager: EntityManager

    def __post_init__(self):
        self.config = self.entity_manager.get('config')
        for output_config in self.config.outputs:
            output_dir = Path(output_config.render_output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

    def update(self):
        """Process audio for all microphones and save to files"""
        frames = self.entity_manager.get('frames')
        current_frame = frames.get()

        mic_done = []
        outputs = self.entity_manager.get('outputs')
        # Find and process audio for each ambisonic output
        for index in outputs.keys():
            if 'AmbisonicOutput' in str(type(outputs[index])):
                output = outputs[index]
                mic_done.append(output.idx)
                for mic in output.get_mics():
                    mic_idx = int(f"{mic.idx*1000}{mic.id}")
                    mic_done.append(mic_idx)
                sample = output.process_audio()
                npz_file = f"{output.output_config.render_output_path}/{output.output_config.idx}.npz"
                _append_to_npz(npz_file, sample)

        # Process audio for each microphone output
        for index in outputs.keys():
            if not index in mic_done:
                output = outputs[index]
                npz_file = f"{output.output_config.render_output_path}/{output.output_config.idx}.npz"
                sample = output.process_audio()
                print('sample', sample, output)
                _append_to_npz(npz_file, sample)
