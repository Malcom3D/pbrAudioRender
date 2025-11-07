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
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numba as nb

#from ..lib.field_ops import calculate_acoustic_energy
from core.entity_manager import EntityManager

@dataclass
class SimulationTermination:
    """Handle simulation termination conditions"""
    entity_manager: EntityManager
    idx: int
    
    def __post_init__(self):
        pass
        
#        self.energy_threshold = getattr(config.wave_propagation, 'termination_energy_threshold', 1e-6)
#        self.max_frames = getattr(config.acoustic_domain, 'sample_frame_limit', 10000)
#        self.min_activity_frames = getattr(config.wave_propagation, 'min_activity_frames', 10)
        
#        self.energy_history = []
#        self.frame_count = 0
#        self.inactive_frames = 0
    
    def update_step(self, layer_manager):
        """Check termination conditions"""
        self.frame_count += 1
        
        # Calculate current energy
        fields = {
            'pressure': layer_manager.pressure,
            'velocity_x': layer_manager.velocity_x,
            'velocity_y': layer_manager.velocity_y,
            'velocity_z': layer_manager.velocity_z
        }
        current_energy = calculate_acoustic_energy(fields)
        
        # Store energy history
        self.energy_history.append(current_energy)
        if len(self.energy_history) > 100:  # Keep last 100 frames
            self.energy_history.pop(0)
        
        # Check for energy decay (simulation has effectively ended)
        if len(self.energy_history) >= self.min_activity_frames:
            recent_energy = np.mean(self.energy_history[-self.min_activity_frames:])
            if recent_energy < self.energy_threshold:
                self.inactive_frames += 1
            else:
                self.inactive_frames = 0
        
        # Check termination conditions
        should_terminate = False
        
        # Condition 1: Maximum frames reached
        if self.frame_count >= self.max_frames:
            should_terminate = True
            print(f"Termination: Reached maximum frame limit ({self.max_frames})")
        
        # Condition 2: Energy below threshold for sufficient time
        elif self.inactive_frames >= self.min_activity_frames:
            should_terminate = True
            print(f"Termination: Energy below threshold for {self.inactive_frames} frames")
        
        # Condition 3: Energy has decayed to negligible levels
        elif current_energy < self.energy_threshold * 0.1:
            should_terminate = True
            print(f"Termination: Energy decayed to negligible level ({current_energy:.2e})")
        
        return should_terminate
    
    def get_termination_stats(self) -> Dict[str, Any]:
        """Get termination statistics"""
        return {
            'frame_count': self.frame_count,
            'inactive_frames': self.inactive_frames,
            'current_energy': self.energy_history[-1] if self.energy_history else 0.0,
            'energy_threshold': self.energy_threshold,
            'max_frames': self.max_frames
        }
    
    def reset(self):
        """Reset termination conditions for new simulation"""
        self.energy_history.clear()
        self.frame_count = 0
        self.inactive_frames = 0

