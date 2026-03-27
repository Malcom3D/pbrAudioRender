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

@dataclass
class AudioRecorder:
    entity_manager: EntityManager

    def __post_init__(self):
        self.config = self.entity_manager.get('config')
        for output_config in self.config.outputs:
            output_dir = output_config.render_output_path
            os.makedirs(output_dir, exist_ok=True)

    def compute(self):
        """Process audio for all microphones and save to files"""
        pass
