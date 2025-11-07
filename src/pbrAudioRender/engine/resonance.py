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
from typing import Dict, List, Tuple, Optional, Any

from ..lib.base import Configurable, GPUEnabled
from .resonances import HelmholtzResonance, ParallelWallResonance, TubeResonance


class Resonance(Configurable, GPUEnabled):
    """Main resonance manager handling all resonant phenomena"""
    
    def __init__(self, config=None, gpu_manager=None):
        super().__init__(config)
        GPUEnabled.__init__(self, gpu_manager)
        
        # Initialize individual resonance handlers
        self.helmholtz = HelmholtzResonance(config, self.gpu)
        self.parallel_wall = ParallelWallResonance(config, self.gpu)
        self.tube = TubeResonance(config, self.gpu)
        
        self.resonance_threshold = config.resonance.resonance_threshold
    
    def detect(self, soxel_grid) -> Dict[str, List[Any]]:
        """Detect resonant structures in the scene"""
        detected_resonances = {
            'helmholtz': [],
            'parallel_wall': [],
            'tube': []
        }
        
        # Delegate detection to individual handlers
        if self.config.resonance.enable_helmholtz:
            detected_resonances['helmholtz'] = self.helmholtz.detect(soxel_grid)
        
        if self.config.resonance.enable_parallel_wall:
            detected_resonances['parallel_wall'] = self.parallel_wall.detect(soxel_grid)
        
        if self.config.resonance.enable_tube:
            detected_resonances['tube'] = self.tube.detect(soxel_grid)
        
        return detected_resonances
    
    def update_step(self, layer_manager, soxel_grid):
        """Apply all resonance effects"""
        updated_layer = layer_manager
        
        # Apply resonances in sequence
        if self.config.resonance.enable_helmholtz:
            updated_layer = self.helmholtz.update(updated_layer, soxel_grid)
        
        if self.config.resonance.enable_tube:
            updated_layer = self.tube.update(updated_layer, soxel_grid)
        
        if self.config.resonance.enable_parallel_wall:
            updated_layer = self.parallel_wall.update(updated_layer, soxel_grid)
        
        return updated_layer
    
    def get_resonance_frequencies(self, soxel_grid) -> Dict[str, List[float]]:
        """Calculate resonance frequencies for detected structures"""
        resonance_freqs = {
            'helmholtz': [],
            'parallel_wall': [],
            'tube': []
        }
        
        detected = self.detect(soxel_grid)
        
        for resonance_type, structures in detected.items():
            for structure in structures:
                if resonance_type == 'helmholtz':
                    freq = self.helmholtz.calculate_resonance_frequency(structure)
                    resonance_freqs['helmholtz'].append(freq)
                elif resonance_type == 'parallel_wall':
                    freq = self.parallel_wall.calculate_resonance_frequency(structure)
                    resonance_freqs['parallel_wall'].append(freq)
                elif resonance_type == 'tube':
                    freq = self.tube.calculate_resonance_frequency(structure)
                    resonance_freqs['tube'].append(freq)
        
        return resonance_freqs

