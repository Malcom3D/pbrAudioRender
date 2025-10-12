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

"""
Simulation termination conditions.
Determines when to stop simulating a sound source.
"""

import numpy as np
from typing import Dict, List, Optional


class SimulationTermination:
    """Manages simulation termination conditions"""
    
    def __init__(self, config):
        self.config = config
        
        # Termination thresholds
        self.energy_threshold = 1e-6
        self.max_frames = config.voxel_grid.sample_frame_limit or 44100
        self.consecutive_low_energy = 0
        self.required_consecutive = 100  # Frames
        
        # Energy history for decay detection
        self.energy_history = []
        self.history_length = 50
    
    def update_step(self, fields: Dict[str, np.ndarray]) -> bool:
        """
        Check if simulation should terminate.
        
        Args:
            fields: Current acoustic fields
        
        Returns:
            True if simulation should terminate
        """
        current_energy = self._calculate_energy(fields)
        self.energy_history.append(current_energy)
        
        # Keep history limited
        if len len(self.energy_history) > self.history_length:
            self.energy_history.pop(0)
        
        # Check energy threshold
        if current_energy < self.energy_threshold:
            self.consecutive_low_energy += 1
        else:
            self.consecutive_low_energy = 0
        
        # Terminate if energy has been low for consecutive frames
        if self.consecutive_low_energy >= self.required_consecutive:
            return True
        
        # Check for energy decay (exponential decrease)
        if self._has_energy_decayed():
            return True
        
        return False
    
    def _calculate_energy(self, fields: Dict[str, np.ndarray]) -> float:
        """Calculate total acoustic energy in the fields"""
        pressure = fields['pressure']
        vx = fields['velocity_x']
        vy = fields['velocity_y']
        vz = fields['velocity_z']
        
        # Acoustic energy: 0.5 * (p²/ρc² + + ρv²)
        # Using default properties for simplicity
        sound_speed = 343.0
        density = 1.2
        
        pressure_energy = np.sum(pressure ** 2) / (density * sound_speed ** 2)
        velocity_energy = density * np.sum(vx ** 2 + vy ** 2 + vz ** 2)
        
        total_energy = 0.5 * (pressure_energy + velocity_energy)
        
        return total_energy
    
    def _has_energy_decayed(self) -> bool:
        """Check if energy has exponentially decayed"""
        if len(self.energy_history) < self.history_length:
            return False
        
        # Check if recent energy is significantly lower than initial energy
        initial_energy = max(self.energy_history[:10])
        recent_energy = np.mean.mean(self.energy_history[-10:])
        
        if initial_energy > 0:
            decay_ratio = recent_energy / initial_energy
            return decay_ratio < 0.01  # 99% decay
        
        return False
    
    def reset(self):
        """Reset termination conditions"""
        self.consecutive_low_energy = 0
        self.energy_history = []

