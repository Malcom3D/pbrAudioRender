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


from pbrAudioCommon.lib.import_helper import np
import numba as nb
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy import ndimage
from sklearn.decomposition import PCA

from ..core.entity_manager import EntityManager
from ..engine.resonances.helmholtz import HelmholtzResonance
from ..engine.resonances.parallel_wall import ParallelWallResonance
from ..engine.resonances.tube import TubeResonance

@dataclass
class Resonance:
    """Main resonance manager handling all resonant phenomena with sophisticated detection"""
    entity_manager: EntityManager
    idx: int
    bands_idx: int
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        
        # Initialize individual resonance handlers
        self.helmholtz = HelmholtzResonance(self.entity_manager, self.idx, self.bands_idx)
        self.parallel_wall = ParallelWallResonance(self.entity_manager, self.idx, self.bands_idx)
        self.tube = TubeResonance(self.entity_manager, self.idx, self.bands_idx)
        
        self.resonance_threshold = config.resonance.resonance_threshold
        self.max_resonance_modes = config.resonance.max_resonance_modes
        self.decay_time_constant = config.resonance.decay_time_constant
        
        # Detection parameters
        self.min_cavity_volume = config.resonance.min_cavity_volume
        self.min_wall_distance = 0.5  # meters
        self.max_wall_distance = 20.0  # meters
        self.min_tube_length = 0.3  # meters
        self.min_aspect_ratio = 3.0  # length/width ratio for tubes
