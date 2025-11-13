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
import numba as nb
from dataclasses import dataclass, field

from core.entity_manager import EntityManager
from lib.interpolate import FrequencyInterpolator

@dataclass
class TransmissionInterface:
    """Handle sound wave transmission through material boundaries with stability control"""
    entity_manager: EntityManager
    idx: int
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.courant_number = config.fdtd.courant_number
        self.max_sound_speed = config.fdtd.max_sound_speed
        self.voxel_size = config.acoustic_domain.voxel_size
        self.dt = 1.0 / config.acoustic_domain.sample_rate
        
        # Transmission parameters
        self.min_sound_speed_ratio = 1.1  # Minimum ratio to trigger transmission handling
        self.max_voxel_multiplier = 4     # Maximum grid refinement for stability
    
    def _check_stability_condition(self, sound_speed: float) -> Tuple[bool, int]:
        """Check Courant stability condition and determine required voxel multiplier"""
        courant_condition = sound_speed * self.dt / self.voxel_size
        
        if courant_condition <= self.courant_number:
            return True, 1  # Stable with current resolution
        
        # Calculate required voxel multiplier to satisfy stability
        required_multiplier = int(np.ceil(courant_condition / self.courant_number))
        required_multiplier = min(required_multiplier, self.max_voxel_multiplier)
        
        return False, required_multiplier
    
    @nb.jit(nopython=True)
    def snells_law(self, incident_angle: float, sound_speed1: float, sound_speed2: float) -> float:
        """Calculate transmission angle using Snell's Law"""
        # Snell's Law: sin(θ1)/c1 = sin(θ2)/c2
        sin_theta2 = (sound_speed2 / sound_speed1) * np.sin(incident_angle)
        
        # Handle total internal reflection
        if abs(sin_theta2) > 1.0:
            return np.pi / 2  # Total internal reflection
        
        return np.arcsin(sin_theta2)
    
    @nb.jit(nopython=True)
    def transmission_coefficient(self, incident_angle: float, impedance1: float, 
                               impedance2: float, sound_speed1: float, sound_speed2: float) -> float:
        """Calculate transmission coefficient using acoustic impedance"""
        # Normal incidence transmission coefficient
        T_normal = 2 * impedance2 / (impedance1 + impedance2)
        
        # Angle-dependent correction (simplified)
        cos_theta1 = np.cos(incident_angle)
        cos_theta2 = np.cos(self.snells_law(incident_angle, sound_speed1, sound_speed2))
        
        # Transmission coefficient for oblique incidence
        T_oblique = T_normal * (cos_theta2 / cos_theta1) if cos_theta1 > 0 else 0.0
        
        return min(1.0, max(0.0, T_oblique))
    
    @nb.jit(nopython=True, parallel=True)
    def apply_transmission_stable(self, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                                sound_speed: np.ndarray, density: np.ndarray, boundaries: Dict,
                                voxel_multipliers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply transmission with stability-controlled voxel resolution"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        impedance = density * sound_speed
        
        for boundary_idx in nb.prange(len(boundaries['sound_speed_discontinuities'])):
            boundary = boundaries['sound_speed_discontinuities'][boundary_idx]
            i, j, k = boundary['position']
            ni, nj, nk = boundary['neighbor_position']
            
            # Get voxel multipliers for stability
            multiplier_current = voxel_multipliers[i, j, k]
            multiplier_neighbor = voxel_multipliers[ni, nj, nk]
            
            # Only process if multipliers indicate transmission handling
            if multiplier_current > 1 or multiplier_neighbor > 1:
                # Get acoustic properties
                c1 = sound_speed[i, j, k]
                c2 = sound_speed[ni, nj, nk]
                Z1 = impedance[i, j, k]
                Z2 = impedance[ni, nj, nk]
                
                # Calculate incident angle from velocity direction
                velocity = np.array([vx[i, j, k], vy[i, j, k], vz[i, j, k]])
                velocity_magnitude = np.sqrt(np.sum(velocity**2))
                
                if velocity_magnitude > 1e-6:
                    velocity_dir = velocity / velocity_magnitude
                    
                    # Estimate surface normal (simplified - pointing from current to neighbor)
                    normal = np.array([ni-i, nj-j, nk-k])
                    normal = normal / np.linalg.norm(normal)
                    
                    incident_angle = np.arccos(np.abs(np.dot(velocity_dir, normal)))
                    
                    # Calculate transmission coefficient
                    T = self.transmission_coefficient(incident_angle, Z1, Z2, c1, c2)
                    
                    # Apply transmission with stability-aware scaling
                    stability_factor = min(multiplier_current, multiplier_neighbor) / self.max_voxel_multiplier
                    transmission_strength = T * stability_factor
                    
                    # Transmit pressure and velocity
                    transmitted_pressure = pressure[i, j, k] * transmission_strength
                    transmitted_velocity = velocity * transmission_strength
                    
                    # Update neighbor voxel (transmitted field)
                    new_pressure[ni, nj, nk] += transmitted_pressure
                    new_vx[ni, nj, nk] += transmitted_velocity[0]
                    new_vy[ni, nj, nk] += transmitted_velocity[1]
                    new_vz[ni, nj, nk] += transmitted_velocity[2]
                    
                    # Reduce energy in current voxel (what's transmitted is removed from incident)
                    new_pressure[i, j, k] -= transmitted_pressure
                    new_vx[i, j, k] -= transmitted_velocity[0]
                    new_vy[i, j, k] -= transmitted_velocity[1]
                    new_vz[i, j, k] -= transmitted_velocity[2]
        
        return new_pressure, new_vx, new_vy, new_vz
    
    def calculate_voxel_multipliers(self, sound_speed: np.ndarray) -> np.ndarray:
        """Calculate required voxel multipliers for stability throughout the domain"""
        multipliers = np.ones(sound_speed.shape, dtype=np.int.int32)
        
        for i in range(sound_speed.shape[0]):
            for j in range(sound_speed.shape[1]):
                for k in range(sound_speed.shape[2]):
                    c = sound_speed[i, j, k]
                    if c > 0:
                        stable, multiplier = self._check_stability_condition(c)
                        multipliers[i, j, k] = multiplier
        
        return multipliers
    
    def update_step(self, layer_manager, soxel_grid, boundaries: Dict):
        """Apply transmission to fields with stability control"""
        config = self.entity_manager.get('config')
        if not config.interface.enable_transmission:
                       return layer_manager
        
        # Get sound speed and density
        sound_speed = soxel_grid.get_array('sound_speed')
        density = soxel_grid.get_array('density')
        
        # Calculate stability multipliers
        voxel_multipliers = self.calculate_voxel_multipliers(sound_speed)
        
        # Apply transmission
        new_pressure, new_vx, new_vy, new_vz = self.apply_transmission_stable(
            layer_manager.get_array('fdtd', 0, 'pressure'),
            layer_manager.get_array('fdtd', 0, 'vx'),
            layer_manager.get_array('fdtd', 0, 'vy'),
            layer_manager.get_array('fdtd', 0, 'vz'),
            sound_speed,
            density,
            boundaries,
            voxel_multipliers
        )
        
        # Update layer manager
        wave_propagpagator = self.entity_manager.get('wave_propagators', self.idx)
        layer_manager = wave_propagator.layer_manager
        layer = layer_manager.get_layer('fdtd', 0)
        
        # Update the layer with new values (simplified - in practice you'd update each FrequencyLimitedField)
        # This would need to be adapted to your specific layer update mechanism
        
        return layer_manager
    
    def get_stability_statistics(self, soxel_grid) -> Dict[str, Any]:
        """Get statistics about stability conditions"""
        sound_speed = soxel_grid.get_array('sound_speed')
        multipliers = self.calculate_voxel_multipliers(sound_speed)
        
        unique_multipliers, counts = np.unique(multipliers, return_counts=True)
        
        stats = {
            'total_voxels': np.prod(sound_speed.shape),
            'stable_voxels': counts[0] if 1 in unique_multipliers else 0,
            'unstable_voxels': np.sum(counts[1:]),
            'max_multiplier_required': np.max(multipliers),
            'multiplier_distribution': dict(zip(unique_multipliers.tolist(), counts.tolist())),
            'courant_number': self.courant_number,
            'max_sound_speed': np.max(sound_speed)
        }
        
        return stats
