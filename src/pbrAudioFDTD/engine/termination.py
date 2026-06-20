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
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numba as nb

from ..core.entity_manager import EntityManager

@dataclass
class SimulationTermination:
    """Handle simulation termination conditions for multi-band, multi-layer architecture"""
    entity_manager: EntityManager
    idx: int
    
    def __post_init__(self):
        # State tracking
        self.frame_count = 0
        self.inactive_frames = 0
        self.energy_history = []
        self.ended = False

    def calculate_acoustic_energy(self) -> float:
        """Calculate total acoustic energy across all layers and frequency bands"""
        total_energy = 0.0
        wave_propagator = self.entity_manager.get('wave_propagators', self.idx)
        layer_manager = wave_propagator.layer_manager
        
        # Get frequency bands for energy weighting
        frequency_bands = self.entity_manager.get('frequency_bands')
        bands = frequency_bands.get_bands()
        
        # Calculate energy for each layer and frequency band
        for layer_idx, layer in layer_manager.layers.items():
            try:
                # Get pressure and velocity arrays for this layer and band
                pressure = layer_manager.get_array(layer.name, layer.bands_idx, 'pressure')
                vx = layer_manager.get_array(layer.name, layer.bands_idx, 'vx')
                vy = layer_manager.get_array(layer.name, layer.bands_idx, 'vy')
                vz = layer_manager.get_array(layer.name, layer.bands_idx, 'vz')
                
                # Calculate energy density: E = 0.5 * (p²/(ρc²) + ρ(vx² + vy² + vz²))
                # Simplified version for relative energy tracking
                energy_density = (
                    np.sum(pressure**2) + 
                    np.sum(vx**2 + vy**2 + vz**2)
                )
                
                # Weight by frequency band (higher frequencies typically have more energy)
                band_center = np.sqrt(bands[layer.bands_idx][0] * bands[layer.bands_idx][1])
                frequency_weight = band_center / 1000.0  # Normalize by 1kHz
                
                total_energy += energy_density * frequency_weight
                
            except (AttributeError, IndexError, KeyError):
                # Skip layers that don't have the required data
                continue
        
        return total_energy

    def update(self) -> bool:
        """Check termination conditions for current simulation state"""
        config = self.entity_manager.get('config')
        wave_propagator = self.entity_manager.get('wave_propagators', self.idx)
        layer_manager = wave_propagator.layer_manager
        
        if layer_manager.ended:
            return True

        # Termination parameters
        termination_config = config.wave_propagation
        termination_energy_threshold = termination_config.termination_energy_threshold
        min_activity_frames = termination_config.min_activity_frames
        max_frames = config.system.frame_limit if config.system.frame_limit else float('inf')

        self.frame_count += 1
        
        # Calculate current energy across all layers and bands
        current_energy = self.calculate_acoustic_energy()
        
        # Store energy history
        self.energy_history.append(current_energy)
        if len(self.energy_history) > 100:  # Keep last 100 frames
            self.energy_history.pop(0)
        
        # Check for energy decay (simulation has effectively ended)
        if len(self.energy_history) >= min_activity_frames:
            recent_energy = np.mean(self.energy_history[-min_activity_frames:])
            if recent_energy < termination_energy_threshold:
                self.inactive_frames += 1
            else:
                self.inactive_frames = 0
        
        # Check termination conditions
        should_terminate = False
        
        # Condition 1: Maximum frames reached
        if self.frame_count >= max_frames:
            should_terminate = True
            print(f"{self.idx} Termination: Reached maximum frame limit ({max_frames})")
        
        # Condition 2: Energy below threshold for sufficient time
        elif self.inactive_frames >= min_activity_frames:
            should_terminate = True
            print(f"{self.idx} Termination: Energy below threshold for {self.inactive_frames} frames")
        
        # Condition 3: Energy has decayed to negligible levels
        elif current_energy < termination_energy_threshold * 0.1:
            should_terminate = True
            print(f"{self.idx} Termination: Energy decayed to negligible level ({current_energy:.2e})")
        
        # Condition 4: All layers have ended (from LayerManager)
        elif hasattr(wave_propagator.layer_manager, 'ended') and wave_propagator.layer_manager.ended:
            should_terminate = True
            print("{self.idx} Termination: All layers have ended")
        
        return should_terminate
