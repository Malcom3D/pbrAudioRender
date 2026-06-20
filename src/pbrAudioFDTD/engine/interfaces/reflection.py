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

from ...core.entity_manager import EntityManager

@dataclass
class ReflectionInterface:
    """Handle sound wave reflection at material boundaries"""
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
    @nb.jit(nopython=True, parallel=True)
    def _apply_reflection(pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, reflection_coeffs: np.ndarray, normal_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply reflection at boundaries using reflection coefficients and surface normals"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        for i in nb.prange(1, pressure.shape[0]-1):
            for j in range(1, pressure.shape[1]-1):
                for k in range(1, pressure.shape[2]-1):
                    R = reflection_coeffs[i, j, k]
                    
                    if R > 0:  # Reflection occurs
                        normal = normal_vectors[i, j, k]
                        
                        # Simple reflection model
                        reflection_strength = R
                        
                        # Reflect pressure (simplified)
                        new_pressure[i, j, k] += pressure[i, j, k] * reflection_strength
                        
                        # Reflect velocity components based on surface normal
                        dot_product = (vx[i, j, k] * normal[0] + 
                                     vy[i, j, k] * normal[1] + 
                                     vz[i, j, k] * normal[2])
                        
                        new_vx[i, j, k] -= 2 * dot_product * normal[0] * reflection_strength
                        new_vy[i, j, k] -= 2 * dot_product * normal[1] * reflection_strength
                        new_vz[i, j, k] -= 2 * dot_product * normal[2] * reflection_strength

        reflected_pressure = new_pressure - pressure
        reflected_vx = new_vx - vx
        reflected_vy = new_vy - vy
        reflected_vz = new_vz - vz
        return reflected_pressure, reflected_vx, reflected_vy, reflected_vz

    def _calculate_surface_normals(self, soxel_types) -> np.ndarray:
        """Calculate surface normals from soxel types using gradient-based approach"""
        normals = np.zeros(soxel_types.shape + (3,), dtype=np.float32)
        
        for i in range(1, soxel_types.shape[0]-1):
            for j in range(1, soxel_types.shape[1]-1):
                for k in range(1, soxel_types.shape[2]-1):
                    # Calculate gradient of soxel types to find surface normals
                    if (soxel_types[i, j, k] == 2 and  # Object boundary
                        (soxel_types[i+1, j, k] == 0 or soxel_types[i-1, j, k] == 0 or
                         soxel_types[i, j+1, k] == 0 or soxel_types[i, j-1, k] == 0 or
                         soxel_types[i, j, k+1] == 0 or soxel_types[i, j, k-1] == 0)):
                        
                        # Calculate gradient using central differences
                        grad_x = (soxel_types[i+1, j, k] - soxel_types[i-1, j, k]) / 2.0
                        grad_y = (soxel_types[i, j+1, k] - soxel_types[i, j-1, k]) / 2.0
                        grad_z = (soxel_types[i, j, k+1] - soxel_types[i, j, k-1]) / 2.0
                        
                        normal = np.array([grad_x, grad_y, grad_z])
                        norm = np.linalg.norm(normal)
                        
                        if norm > 0:
                            normals[i, j, k] = normal / norm
                        else:
                            normals[i, j, k] = np.array([0.0, 0.0, 0.0])
        
        return normals

    def update(self, boundaries: Dict[str, Any]):
        """Apply reflection to fields at material boundaries"""
        config = self.entity_manager.get('config')
        enable_reflection = config.interface.enable_reflection
        if not enable_reflection:
            return

        voxel_volume = config.acoustic_domain.voxel_size**3
        soxel_grid = self.entity_manager.get('soxel_grid')
        wave_propagator = self.entity_manager.get('wave_propagators', self.idx)
        layer_manager = wave_propagator.layer_manager

        # Get reflection coefficients and surface normals
        reflection_coeffs = soxel_grid.get_array('reflection', self.low_freq, self.high_freq)
        soxel_types = soxel_grid.get_array('type')
        surface_normals = self._calculate_surface_normals(soxel_types)

        names = []
        items = list(layer_manager.layers.items())
        for index, item in items:
            if not item.name in names and item.bands_idx == self.bands_idx:
                name = item.name
                names.append(name)

                # Apply reflection
                reflected_pressure, reflected_vx, reflected_vy, reflected_vz = self._apply_reflection(
                    layer_manager.get_array(name, self.bands_idx, 'pressure'),
                    layer_manager.get_array(name, self.bands_idx, 'vx'),
                    layer_manager.get_array(name, self.bands_idx, 'vy'),
                    layer_manager.get_array(name, self.bands_idx, 'vz'),
                    reflection_coeffs,
                    surface_normals
                )

                # Apply (reflected_pressure, reflected_vx, reflected_vy, reflected_vz) to selected reflection layer
                if name == 'fdtd':
                    layer_manager.add_new('reflection_0', self.bands_idx)
                    reflection_layer = layer_manager.get_layer('reflection_0', self.bands_idx)
                    reflection_order = 0
                elif 'reflection' in name:
                    reflection_order = int(layer_manager.layers[index].name.removeprefix('reflection_')) + 1 
                    layer_manager.add_new(f"reflection_{reflection_order}", self.bands_idx)
                    reflection_layer = layer_manager.get_layer(f"reflection_{reflection_order}", self.bands_idx)

                reflection_pressure = layer_manager.get_array(f"reflection_{reflection_order}", self.bands_idx, 'pressure')
                reflection_vx = layer_manager.get_array(f"reflection_{reflection_order}", self.bands_idx, 'vx')
                reflection_vy = layer_manager.get_array(f"reflection_{reflection_order}", self.bands_idx, 'vy')
                reflection_vz = layer_manager.get_array(f"reflection_{reflection_order}", self.bands_idx, 'vz')

                _pressure = (reflected_pressure*voxel_volume + reflection_pressure*voxel_volume)/(2*voxel_volume)
                _vx = (reflected_vx*voxel_volume + reflection_vx*voxel_volume)/(2*voxel_volume)
                _vy = (reflected_vy*voxel_volume + reflection_vy*voxel_volume)/(2*voxel_volume)
                _vz = (reflected_vz*voxel_volume + reflection_vz*voxel_volume)/(2*voxel_volume)

                layer_manager.update_layer(reflection_layer.name, reflection_layer.bands_idx, self.low_freq, self.high_freq, _pressure, _vx, _vy, _vz)
