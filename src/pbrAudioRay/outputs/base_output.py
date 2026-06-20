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

import json
from pbrAudioCommon.lib.import_helper import np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.interpolator import FrequencyInterpolator

@dataclass
class BaseOutput:
    """Base class for all microphone outputs."""
    entity_manager: EntityManager
    config_idx: int  # idx of the output in config.outputs list
    
    def __post_init__(self):
        self.config = self.entity_manager.get('config')
        # Find output config
        self.output_config = None
        for oc in self.config.outputs:
            if oc.idx == self.config_idx:
                self.output_config = oc
                break
        self.sample_rate = self.config.system.sample_rate
        self.subframes = self.config.system.subframes
        self.fps = self.config.system.fps
    
#    def get_impulse_response(self, source_idx: int):
#        """Retrieve the impulse response computed for this output and given source."""
#        # The impulse response is stored in the wave propagator for this source-output pair
#        # The propagator is registered as 'wave_propagators'
#        wave_propagator = self.entity_manager.get('wave_propagators')
#        if wave_propagator is None:
#            raise RuntimeError("No wave propagators found")
#        # Find propagator for this source and output
#        for wp in wave_propagator.values():
#            if wp.source_idx == source_idx and wp.output_idx == self.config_idx:
#                return wp.get_impulse_response()
#        raise RuntimeError(f"No wave propagator for source {source_idx} and output {self.config_idx}")
    
    def record_audio(self, source_audio: np.ndarray, source_idx: int, output_file: str):
        """Convolve source audio with impulse response and save."""
        ir_time, ir_amp = self.get_impulse_response(source_idx)
        # Resample impulse response to sample rate (already at sample rate)
        # Convolve
        convolved = np.convolve(source_audio, ir_amp, mode='full')
        # Trim to length of source audio? Usually we keep full
        # Save
        import soundfile as sf
        sf.write(output_file, convolved, self.sample_rate)
    
    def _get_directivity(self, azimuth: float, elevation: float) -> float:
        """Base directivity pattern - should be overridden by subclasses"""
        return 1.0  # Omnidirectional by default
