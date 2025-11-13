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
from dataclasses import dataclass, field

from core.entity_manager import EntityManager
from lib.interpolate import FrequencyInterpolator, SpatialInterpolator
from .interfaces import (
    AbsorptionInterface, ReflectionInterface, TransmissionInterface,
    ScatteringInterface, DiffractionInterface
)

@dataclass
class InterfaceManager:
    """Main interface manager handling all boundary interactions with sophisticated detection"""
    entity_manager: EntityManager
    idx: int
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        
        # Initialize individual interface handlers
        self.absorption = AbsorptionInterface(self.entity_manager, self.idx)
        self.reflection = ReflectionInterface(self.entity_manager, self.idx)
        self.transmission = TransmissionInterface(self.entity_manager, self.idx)
        self.scattering = ScatteringInterface(self.entity_manager, self.idx)
        self.diffraction = DiffractionInterface(self.entity_manager, self.idx)
        
        self.interaction_threshold = config.interface.interaction_threshold
        self.min_impedance_ratio = config.interface.min_impedance_ratio
        self.max_impedance_ratio = config.interface.max_impedance_ratio
        
        # Layer management for reflections and scattering
        self.reflection_layers = {}
        self.scattering_layers = {}
        self.max_reflections = config.acoustic_domain.max_reflections
    
    def detect_boundaries(self, layer_manager, soxel_grid, frequency: float = 1000.0) -> Dict[str, Any]:
        """Sophisticated boundary detection using impedance discontinuities"""
        boundaries = {
            'impedance_discontinuities': [],
            'sound_speed_discontinuities': [],
            'material_boundaries': [],
            'edge_boundaries': []
        }
        
        # Get acoustic properties
        sound_speed = soxel_grid.get_array('sound_speed')
        density = soxel_grid.get_array('density')
        
        # Calculate acoustic impedance Z = ρc
        impedance = density * sound_speed
        
        pressure = layer_manager.get_array('fdtd', 0, 'pressure')  # Get pressure from first band
        
        for i in range(1, impedance.shape[0]-1):
            for j in range(1, impedance.shape[1]-1):
                for k in range(1, impedance.shape[2]-1):
                    current_impedance = impedance[i, j, k]
                    
                    # Check neighbors for impedance discontinuities
                    for di, dj, dk in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                        ni, nj, nk = i + di, j + dj, k + dk
                        
                        if (0 <= ni < impedance.shape[0] and 
                            0 <= nj < impedance.shape[1] and 
                            0 <= nk < impedance.shape[2]):
                            
                            neighbor_impedance = impedance[ni, nj, nk]
                            
                            # Calculate impedance ratio
                            if current_impedance > 0 and neighbor_impedance > 0:
                                impedance_ratio = max(current_impedance, neighbor_impedance) / min(current_impedance, neighbor_impedance)
                                
                                # Check if this is a significant discontinuity
                                if (impedance_ratio > self.min_impedance_ratio and 
                                    impedance_ratio < self.max_impedance_ratio and
                                    np.abs(pressure[i, j, k]) > self.interaction_threshold):
                                    
                                    boundaries['impedance_discontinuities'].append({
                                        'position': (i, j, k),
                                        'neighbor_position': (ni, nj, nk),
                                        'impedance_ratio': impedance_ratio,
                                        'pressure': pressure[i, j, k],
                                        'type': 'impedance'
                                    })
                            
                            # Check sound speed discontinuities for transmission
                            current_sound_speed = sound_speed[i, j, k]
                            neighbor_sound_speed = sound_speed[ni, nj, nk]
                            
                            if current_sound_speed > 0 and neighbor_sound_speed > 0:
                                sound_speed_ratio = max(current_sound_speed, neighbor_sound_speed) / min(current_sound_speed, neighbor_sound_speed)
                                
                                if sound_speed_ratio > 1.1:  # Significant sound speed change
                                    boundaries['sound_speed_discontinuities'].append({
                                        'position': (i, j, k),
                                        'neighbor_position': (ni, nj, nk),
                                        'sound_speed_ratio': sound_speed_ratio,
                                        'type': 'sound_speed'
                                    })
        
        return boundaries
    
    def should_apply_interfaces(self, layer_manager, soxel_grid) -> bool:
        """Determine if interface interactions should be applied"""
        boundaries = self.detect_boundaries(layer_manager, soxel_grid)
        
        # Apply interfaces if we have significant boundaries and sufficient pressure
        pressure = layer_manager.get_array('fdtd', 0, 'pressure')
        max_pressure = np.max(np.abs(pressure))
        
        return (len(boundaries['impedance_discontinuities']) > 0 or 
                len(boundaries['sound_speed_discontinuities']) > 0) and \
               max_pressure > self.interaction_threshold
    
    def update_step(self, layer_manager, soxel_grid):
        """Apply all interface interactions with proper layer management"""
        if not self.should_apply_interfaces(layer_manager, soxel_grid):
            return layer_manager
        
        boundaries = self.detect_boundaries(layer_manager, soxel_grid)
        updated_layer = layer_manager
        
        # Apply interface interactions in physically correct order
        
        # 1. Diffraction (occurs at edges before other interactions)
        if boundaries['edge_boundaries']:
            updated_layer = self.diffraction.update_step(updated_layer, soxel_grid, boundaries)
        
        # 2. Transmission (wave propagation through boundaries)
        if boundaries['sound_speed_discontinuities']:
            updated_layer = self.transmission.update_step(updated_layer, soxel_grid, boundaries)
        
        # 3. Absorption (energy loss at boundaries)
        if boundaries['impedance_discontinuities']:
            updated_layer = self.absorption.update_step(updated_layer, soxel_grid, boundaries)
        
        # 4. Reflection and Scattering with layer management
        if boundaries['impedance_discontinuities']:
            updated_layer = self._apply_reflection_scattering_layers(updated_layer, soxel_grid, boundaries)
        
        return updated_layer
    
    def _apply_reflection_scattering_layers(self, layer_manager, soxel_grid, boundaries):
        """Apply reflection and scattering with proper layer management"""
        current_pressure = layer_manager.get_array('fdtd', 0, 'pressure')
        
        # Apply primary reflection
        reflected_layer = self.reflection.update_step(layer_manager, soxel_grid, boundaries)
        
        # Store primary reflection in reflection layer
        reflection_pressure = reflected_layer.get_array('fdtd', 0, 'pressure') - current_pressure
        self._update_reflection_layer(0, reflection_pressure)
        
        # Apply scattering
        scattered_layer = self.scattering.update_step(reflected_layer, soxel_grid, boundaries)
        
        # Store scattering in scattering layer
        scattering_pressure = scattered_layer.get_array('fdtd', 0, 'pressure') - reflected_layer.get_array('fdtd', 0, 'pressure')
        self._update_scattering_layer(0, scattering_pressure)
        
        # Apply higher order reflections if enabled
        if self.max_reflections > 1:
            scattered_layer = self._apply_higher_order_reflections(scattered_layer, soxel_grid, boundaries)
        
        return scattered_layer
    
    def _update_reflection_layer(self, order: int, pressure: np.ndarray):
        """Update reflection layer for given order"""
        if order not in self.reflection_layers:
            self.reflection_layers[order] = np.zeros_like(pressure)
        self.reflection_layers[order] += pressure
    
    def _update_scattering_layer(self, order: int, pressure: np.ndarray):
        """Update scattering layer for given order"""
        if order not in self.scattering_layers:
            self.scattering_layers[order] = np.zeros_like(pressure)
        self.scattering_layers[order] += pressure
    
    def _apply_higher_order_reflections(self, layer_manager, soxel_grid, boundaries, current_order: int = 1):
        """Apply higher order reflections recursively"""
        if current_order >= self.max_reflections:
            return layer_manager
        
        # Calculate energy of current reflection to see if we should continue
        current_pressure = layer_manager.get_array('fdtd', 0, 'pressure')
        reflection_energy = np.sum(current_pressure ** 2)
        
        if reflection_energy < self.interaction_threshold * 0.1:
            return layer_manager
        
        # Apply next order reflection
        next_reflection = self.reflection.update_step(layer_manager, soxel_grid, boundaries)
        reflection_pressure = next_reflection.get_array('fdtd', 0, 'pressure') - current_pressure
        
        self._update_reflection_layer(current_order, reflection_pressure)
        
        # Recursively apply higher orders
        return self._apply_higher_order_reflections(next_reflection, soxel_grid, boundaries, current_order + 1)
    
    def get_interface_statistics(self, layer_manager, soxel_grid) -> Dict[str, Any]:
        """Get detailed statistics about interface interactions"""
        boundaries = self.detect_boundaries(layer_manager, soxel_grid)
        
        stats = {
            'boundaries_detected': sum(len(b) for b in boundaries.values()),
            'impedance_discontinuities': len(boundaries['impedance_discontinuities']),
            'sound_speed_discontinuities': len(boundaries['sound_speed_discontinuities']),
            'material_boundaries': len(boundaries['material_boundaries']),
            'edge_boundaries': len(boundaries['edge_boundaries']),
            'reflection_layers': len(self.reflection_layers),
            'scattering_layers': len(self.scattering_layers),
            'max_reflection_order': max(self.reflection_layers.keys()) if self.reflection_layers else 0,
            'interaction_threshold': self.interaction_threshold
        }
        
        # Add energy statistics for each layer
        for order, layer_pressure in self.reflection_layers.items():
            stats[f'reflection_layer_{order}_energy'] = np.sum(layer_pressure ** 2)
        
        for order, layer_pressure in self.scattering_layers.items():
            stats[f'scattering_layer_{order}_energy'] = np.sum(layer_pressure ** 2)
        
        return stats
    
    def reset_layers(self):
        """Reset reflection and scattering layers"""
        self.reflection_layers.clear()
        self.scattering_layers.clear()

