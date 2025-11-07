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
from ..lib.interpolate import FrequencyInterpolator, SpatialInterpolator
from .interfaces import (
    AbsorptionInterface, ReflectionInterface, RefractionInterface,
    ScatteringInterface, DiffractionInterface
)


class Interface(Configurable, GPUEnabled):
    """Main interface manager handling all boundary interactions"""
    
    def __init__(self, config=None, gpu_manager=None):
        super().__init__(config)
        GPUEnabled.__init__(self, gpu_manager)
        
        # Initialize individual interface handlers
        self.absorption = AbsorptionInterface(config, self.gpu)
        self.reflection = ReflectionInterface(config, self.gpu)
        self.refraction = RefractionInterface(config, self.gpu)
        self.scattering = ScatteringInterface(config, self.gpu)
        self.diffraction = DiffractionInterface(config, self.gpu)
        
        self.interaction_threshold = config.interface.interaction_threshold
    
    def detect(self, layer_manager, soxel_grid) -> bool:
        """Detect if fields interact with material boundaries"""
        pressure = layer_manager.pressure
        max_pressure = np.max(np.abs(pressure))
        
        # Simple detection based on pressure magnitude
        if max_pressure < self.interaction_threshold:
            return False
        
        # More sophisticated detection could check for impedance discontinuities
        # For now, assume interactions occur when pressure is above threshold
        return True
    
    def update_step(self, layer_manager, soxel_grid):
        """Apply all interface interactions"""
        if not self.detect(layer_manager, soxel_grid):
            return layer_manager
        
        updated_layer = layer_manager
        
        # Apply interface interactions in sequence
        # Note: The order matters and should follow physical principles
        
        # 1. Diffraction (occurs at edges before other interactions)
        updated_layer = self.diffraction.update_step(updated_layer, soxel_grid)
        
        # 2. Absorption (energy loss at boundaries)
        updated_layer = self.absorption.update_step(updated_layer, soxel_grid)
        
        # 3. Refraction (wave bending at boundaries)
        updated_layer = self.refraction.update_step(updated_layer, soxel_grid)
        
        # 4. Reflection (wave bouncing at boundaries)
        updated_layer = self.reflection.update_step(updated_layer, soxel_grid)
        
        # 5. Scattering (diffuse reflection)
        updated_layer = self.scattering.update_step(updated_layer, soxel_grid)
        
        return updated_layer
    
    def get_interaction_statistics(self, layer_manager, soxel_grid) -> Dict[str, float]:
        """Get statistics about interface interactions"""
        stats = {}
        
        if self.detect(layer_manager, soxel_grid):
            pressure = layer_manager.pressure
            stats.update({
                'max_pressure': np.max(np.abs(pressure)),
                'interaction_detected': True,
                'active_interfaces': self._get_active_interfaces()
            })
        else:
            stats.update({
                'max_pressure': np.max(np.abs(layer_manager.pressure)),
                'interaction_detected': False,
                'active_interfaces': []
            })
        
        return stats
    
    def _get_active_interfaces(self) -> List[str]:
        """Get list of active interface handlers handlers"""
        active = []
        if self.config.interface.diffraction_enabled:
            active.append('diffraction')
        if self.config.interface.absorption_enabled:
            active.append('absorption')
        if self.config.interface.refraction_enabled:
            active.append('refraction')
        if self.config.interface.reflection_enabled:
            active.append('reflection')
        if self.config.interface.scattering_enabled:
            active.append('scattering')
        
        return active

