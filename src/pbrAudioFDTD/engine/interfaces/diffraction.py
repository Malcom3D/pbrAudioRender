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

from pbrAudioCommon import np
import numba as nb
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ...core.entity_manager import EntityManager

@nb.jit(nopython=True)
def _utd_diffraction_coefficient(incident_angle: float, diffraction_angle: float, center_freq: float, obstacle_size: float, sound_speed: float) -> complex:
    """Calculate UTD diffraction coefficient using Uniform Theory of Diffraction"""
    k = 2 * np.pi * center_freq / sound_speed  # wave number
    
    # Edge diffraction parameters
    L = obstacle_size
    n = 2.0  # Wedge angle parameter (n=2 for 90° wedge)
    
    # Angles relative to wedge faces
    phi = incident_angle
    phi_prime = diffraction_angle
    
    # UTD diffraction coefficient components
    # F(x) = 2j√(x)exp(jx)∫_√x^∞ exp(-jτ²)dτ (Fresnel integral)
    
    # For simplified implementation, we'll use approximation
    # This is a simplified version - full UTD implementation would be more complex
    
    if L == 0:
        return 0.0 + 0.0j
    
    # Parameter for Fresnel zone
    X = k * L * (np.sin(phi) + np.sin(phi_prime))**2
    
    # Fresnel integral approximation
    if X < 0.1:
        F = 1.0
    else:
        F = np.sqrt(np.pi * X) * np.exp(1j * (X + np.pi/4))
    
    # Diffraction coefficient (simplified UTD)
    D = -np.exp(-1j * np.pi/4) / (2 * n * np.sqrt(2 * np.pi * k)) * (np.tan((phi - phi_prime) / 2) + np.tan((phi + phi_prime) / 2)) * F
    return D

@dataclass
class DiffractionInterface:
    """Handle sound wave diffraction around obstacles using UTD model"""
    entity_manager: EntityManager
    idx: int
    bands_idx: int

    def __post_init__(self):
        # Get low and high frequency
        frequency_bands = self.entity_manager.get('frequency_bands')
        bands = frequency_bands.get_bands()
        self.low_freq = bands[self.bands_idx][0]
        self.high_freq = bands[self.bands_idx][1]

    @staticmethod
#    @nb.jit(nopython=True, parallel=True)
    def _apply_diffraction(pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, soxel_types: np.ndarray, boundaries: Dict, center_freq: float, sound_speed: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply UTD diffraction to fields at edge boundaries"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        # Process edge boundaries
        for boundary_idx in nb.prange(len(boundaries.get('edge_boundaries', []))):
            boundary = boundaries['edge_boundaries'][boundary_idx]
            i, j, k = boundary['position']
            obstacle_size = boundary['obstacle_size']
            
            edge_pressure = pressure[i, j, k]
            
            if np.abs(edge_pressure) > 1e-6:
                # Calculate incident wave direction from velocity vectors
                incident_direction = np.array([vx[i, j, k], vy[i, j, k], vz[i, j, k]])
                incident_magnitude = np.linalg.norm(incident_direction)
                
                if incident_magnitude > 1e-6:
                    incident_direction = incident_direction / incident_magnitude
                        
                    # For each diffraction direction (simplified - sample key directions)
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                        
                                ni, nj, nk = i + di, j + dj, k + dk
                                
                                if (0 <= ni < pressure.shape[0] and 
                                    0 <= nj < pressure.shape[1] and 
                                    0 <= nk < pressure.shape[2]):
                                    
                                    # Check if neighbor is in shadow region (free space)
                                    if soxel_types[ni, nj, nk] == 0:
                                        # Calculate diffraction direction
                                        diffract_direction = np.array([di, dj, dk])
                                        diffract_direction = diffract_direction / np.linalg.norm(diffract_direction)
                                        
                                        # Calculate angles for UTD
                                        incident_angle = np.arccos(np.dot(incident_direction, diffract_direction))
                                        diffraction_angle = np.pi - incident_angle  # For back diffraction
                                        
                                        # Apply UTD diffraction coefficient
                                        D = _utd_diffraction_coefficient(incident_angle, diffraction_angle, center_freq, obstacle_size, sound_speed[ni, nj, nk])
                                        
                                        # Calculate diffracted field
                                        distance = np.sqrt(di**2 + dj**2 + dk**2) * voxel_size
                                        if distance > 0:
                                            # Spreading factor and phase term
                                            spreading = 1.0 / np.sqrt(distance)
                                            phase_term = np.exp(1j * k * distance)
                                            
                                            # Diffracted pressure
                                            diffracted_pressure = (edge_pressure * D * spreading * phase_term)
                                            
                                            # Apply to neighbor (real part for acoustic simulation)
                                            new_pressure[ni, nj, nk] += np.real(diffracted_pressure)
    
        return new_pressure, new_vx, new_vy, new_vz
    
    def update(self, boundaries: Dict[str, Any]):
        """Apply diffraction to fields using UTD model"""
        config = self.entity_manager.get('config')
        enable_diffraction = config.interface.enable_diffraction
        if not enable_diffraction:
            return

        center_freq = (self.low_freq + self.high_freq)/2

        voxel_size = config.acoustic_domain.voxel_size
        soxel_grid = self.entity_manager.get('soxel_grid')
        sound_speed = soxel_grid.get_array('sound_speed')
        soxel_types = soxel_grid.get_array('type')
        wave_propagator = self.entity_manager.get('wave_propagators', self.idx)
        layer_manager = wave_propagator.layer_manager

        names = []
        items = list(layer_manager.layers.items())
        for index, item in items:
            if not item.name in names and item.bands_idx == self.bands_idx:
                name = item.name
                names.append(name)

                # Apply diffraction
                new_pressure, new_vx, new_vy, new_vz = self._apply_diffraction(
                    layer_manager.get_array(name, self.bands_idx, 'pressure'),
                    layer_manager.get_array(name, self.bands_idx, 'vx'),
                    layer_manager.get_array(name, self.bands_idx, 'vy'),
                    layer_manager.get_array(name, self.bands_idx, 'vz'),
                    soxel_types,
                    boundaries,
                    center_freq,
                    sound_speed,
                    voxel_size
                )

                # Apply (new_pressure, new_vx, new_vy, new_vz) to selected layer
                layer = layer_manager.get_layer(name, self.bands_idx)
                layer_manager.update_layer(layer.name, layer.bands_idx, self.low_freq, self.high_freq, new_pressure, new_vx, new_vy, new_vz)
