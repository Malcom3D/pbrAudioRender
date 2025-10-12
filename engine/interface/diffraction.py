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
Diffraction handling around obstacles and edges.
"""

"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import numba as nb

from ...utils.parallel_proc import configure_numba


class DiffractionManager:
    """Manages sound wave diffraction around obstacles"""
    
    def __init__(self, config, gpu_manager=None):
        self.config = config
        self.gpu = gpu_manager
        self.jit = configure_numba(parallel=True)
    
    def apply_diffraction(self, fields: Dict[str, np.ndarray],
                         boundaries: List[Dict[str, Any]],
                         soxel_grid) -> Dict[str, np.ndarray]:
        """
        Apply diffraction around detected boundaries and edges.
        
        Args:
            fields: Current acoustic fields
            boundaries: Detected boundary information
            soxel_grid: SoxelGrid for material properties
        
        Returns:
            Fields with diffraction applied
        """
        if not boundaries:
            return fields
        
        result_fields = fields.copy()
        
        if self.gpu and self.gpu.config.use_gpu:
            result_fields = self._apply_diffraction_gpu(result_fields, boundaries, soxel_grid)
        else:
            result_fields = self._apply_diffraction_cpu(result_fields, boundaries, soxel_grid)
        
        return result_fields
    
    def _apply_diffraction_cpu(self, fields: Dict[str, np.ndarray],
                             boundaries: List[Dict[str, Any]],
                             soxel_grid) -> Dict[str, np.ndarray]:
        """CPU implementation of diffraction"""
        pressure = fields['pressure'].copy()
        vx = fields['velocity_x'].copy()
        vy = fields['velocity_y'].copy()
        vz = fields['velocity_z'].copy()
        
        shape = pressure.shape
        
        # Find edges where diffraction occurs (boundaries with specific geometry)
        diffraction_edges = self._find_diffraction_edges(boundaries, soxel_grid)
        
        for edge in diffraction_edges:
            edge_position = edge['position']
            i, j, k = edge_position
            
            # Calculate diffraction based on wavelength and geometry
            wavelength = self._calculate_wavelength(soxel_grid.grid[i, j, k])
            diffraction_strength = self._calculate_diffraction_strength(
                edge, wavelength
            )
            
            if diffraction_strength > 0:
                # Apply diffraction by spreading energy around the edge
                diffracted_energy = pressure[i, j, k] * diffraction_strength
                
                # Find voxels in the "shadow region" behind the edge
                shadow_voxels = self._find_shadow_voxels(edge_position, edge['normal'], shape)
                
                if shadow_voxels:
                    energy_per_voxel = diffracted_energy / len(shadow_voxels)
                    
                    for vi, vj, vk in shadow_voxels:
                        pressure[vi, vj, vk] += energy_per_voxel
        
        return {
            'pressure': pressure,
            'velocity_x': vx,
            'velocity_y': vy,
            'velocity_z': vz
        }
    
    def _find_diffraction_edges(self, boundaries: List[Dict[str, Any]],
                              soxel_grid) -> List[Dict[str, Any]]:
        """Find edges where diffraction is likely to occur"""
        edges = []
        
        for boundary in boundaries:
            i, j, k = boundary['position']
            di, dj, dk = boundary['direction']
            
            # Simple edge detection: look for boundaries with specific neighbor patterns
            if self._is_edge_voxel((i, j, k), soxel_grid):
                edge_info = {
                    'position': (i, j, k),
                    'normal': (di, dj, dk),
                    'boundary': boundary
                }
                edges.append(edge_info)
        
        return edges
    
    def _is_edge_voxel(self, position: Tuple[int, int, int],
                      soxel_grid) -> bool:
        """Check if a voxel is at an edge where diffraction occurs"""
        i, j, k = position
        shape = soxel_grid.shape
        
        # Count solid neighbors (non-default medium)
        solid_neighbors = 0
        total_neighbors = 0
        
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    
                    ni, nj, nk = i + di, j + dj, k + dk
                    
                    if (0 <= ni < shape[0] and 
                        0 <= nj < shape[1] and 
                        0 <= nk < shape[2]):
                        total_neighbors += 1
                        neighbor_soxel = soxel_grid.grid[ni, nj, nk]
                        
                        # Consider it solid if not default medium
                        if neighbor_soxel.idx != 0:
                            solid_neighbors += 1
        
        # Consider it an edge if it has both solid and non-solid neighbors
        return 0 < solid_neighbors < total_neighbors
    
    def _calculate_wavelength(self, soxel, frequency: float = 1000.0) -> float:
        """Calculate wavelength for given frequency"""
        return soxel.sound_speed / frequency
    
    def _calculate_diffraction_strength(self, edge: Dict[str, Any],
                                      wavelength: float) -> float:
        """Calculate diffraction strength based on edge geometry and wavelength"""
        # Simplified diffraction model
        # In practice, use more sophisticated models like UTD
        base_strength = 0.1
        
        # Wavelength-dependent diffraction
        # Higher frequencies (shorter wavelengths) diffract less
        wavelength_factor = 1.0 / (1.0 + wavelength)
        
        return base_strength * wavelength_factor
    
    def _find_shadow_voxels(self, edge_position: Tuple[int, int, int],
                          normal: Tuple[float, float, float],
                          shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Find voxels in the shadow region behind an edge"""
        i, j, k = edge_position
        shadow_voxels = []
        
        # Simple shadow region: voxels behind the edge in the normal direction
        steps = 3  # Look 3 voxels into shadow region
        
        for step in range(1, steps + 1):
            ni = i + int(normal[0] * step)
            nj = j + int(normal[1] * step)
            nk = k + int(normal[2] * step)
            
            if (0 <= ni < shape[0] and 
                0 <= nj < shape[1] and 
                0 <= nk < shape[2]):
                shadow_voxels.append((ni, nj, nk))
        
        return shadow_voxels
    
    def _apply_diffraction_gpu(self, fields: Dict[str, np.ndarray],
                             boundaries: List[Dict[str, Any]],
                             soxel_grid) -> Dict[str, np.ndarray]:
        """GPU implementation of diffraction"""
        # For now, fall back to CPU implementation
        return self._apply_diffraction_cpu(fields, boundaries, soxel_grid)

