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

from ...lib.base import Configurable, GPUEnabled


class HelmholtzResonance(Configurable, GPUEnabled):
    """Handle Helmholtz resonator effects"""
    
    def __init__(self, config=None, gpu_manager=None):
        super().__init__(config)
        GPUEnabled.__init__(self, gpu_manager)
        
        self.min_cavity_volume = config.resonance.min_cavity_volume
        self.max_resonance_modes = config.resonance.max_resonance_modes
    
    def detect(self, soxel_grid) -> List[Dict]:
        """Detect Helmholtz resonators in the scene"""
        resonators = []
        
        # Simple detection based on cavity and neck structure
        # In practice, use more sophisticated cavity detection
        for i in range(1, soxel_grid.shape[0]-1):
            for j in range(1, soxel_grid.shape[1]-1):
                for k in range(1, soxel_grid.shape[2]-1):
                    if self._is_helmholtz_resonator(soxel_grid, i, j, k):
                        resonator = {
                            'position': (i, j, k),
                            'cavity_volume': self._estimate_cavity_volume(soxel_grid, i, j, k),
                            'neck_dimensions': self._estimate_neck_dimensions(soxel_grid, i, j, k)
                        }
                        resonators.append(resonator)
        
        return resonators
    
    def _is_helmholtz_resonator(self, soxel_grid, i: int, j: int, k: int) -> bool:
        """Check if position contains a Helmholtz resonator"""
        # Simplified detection - look for small openings in larger cavities
        if soxel_grid.soxel_types[i, j, k] != 2:  # Not an object
            return False
        
        # Check for cavity-like structure
        cavity_size = self._count_connected_voxels(soxel_grid, i, j, k)
        
        if cavity_size < 8:  # Minimum cavity size
            return False
        
        # Check for neck-like structure (small opening)
        neck_found = self._find_neck_opening(soxel_grid, i, j, k)
        
        return neck_found is not None
    
    def _count_connected_voxels(self, soxel_grid, i: int, j: int, k: int) -> int:
        """Count connected object voxels (simplified flood fill)"""
        visited = set()
        stack = [(i, j, k)]
        count = 0
        
        while stack and count < 1000:  # Limit search
            x, y, z = stack.pop()
            if (x, y, z) in visited:
                continue
            
            visited.add((x, y, z))
            count += 1
            
            # Check neighbors
            for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                if (0 <= nx < soxel_grid.shape[0] and 
                    0 <= ny < soxel_grid.shape[1] and 
                    0 <= nz < soxel_grid.shape[2] and
                    soxel_grid.soxel_types[nx, ny, nz] == 2):
                    stack.append((nx, ny, nz))
        
        return count
    
    def _find_neck_opening(self, soxel_grid, i: int, j: int, k: int) -> Optional[Tuple[int, int, int]]:
        """Find neck opening in cavity"""
        # Look for small openings to the exterior
        for di, dj, dk in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            ni, nj, nk = i + di, j + dj, k + dk
            
            if (0 <= ni < soxel_grid.shape[0] and 
                0 <= nj < soxel_grid.shape[1] and 
                0 <= nk < soxel_grid.shape[2] and
                soxel_grid.soxel_types[ni, nj, nk] != 2):  # Opening to non-object
                
                # Check if this is a small opening
                opening_size = self._measure_opening_size(soxel_grid, ni, nj, nk)
                if opening_size < 4:  # Small opening
                    return (ni, nj, nk)
        
        return None
    
    def _measure_opening_size(self, soxel_grid, i: int, j: int, k: int) -> int:
        """Measure size of opening"""
        # Simplified - count connected non-object voxels
        return 1  # Placeholder
    
    def _estimate_cavity_volume(self, soxel_grid, i: int, j: int, k: int) -> float:
        """Estimate cavity volume in cubic meters"""
        voxel_volume = soxel_grid.voxel_size ** 3
        cavity_voxels = self._count_connected_voxels(soxel_grid, i, j, k)
        return cavity_voxels * voxel_volume
    
    def _estimate_neck_dimensions(self, soxel_grid, i: int, j: int, k: int) -> Dict[str, float]:
        """Estimate neck dimensions"""
        neck = self._find_neck_opening(soxel_grid, i, j, k)
        if neck:
            return {
                'length': soxel_grid.voxel_size,
                'area': soxel_grid.voxel_size ** 2,
                'position': neck
            }
        return {'length': 0, 'area': 0, 'position': None}
    
    def calculate_resonance_frequency(self, resonator: Dict) -> float:
        """Calculate Helmholtz resonance frequency"""
        # f = (c/2π) * √(A/(V*L))
        c = 343.0  # sound speed
        V = resonator['cavity_volume']
        neck = resonator['neck_dimensions']
        
        if V <= 0 or neck['area'] <= 0:
            return 0.0
        
        A = neck['area']
        L = neck['length']
        
        # Effective neck length correction
        L_eff = L + 0.8 * np.sqrt(A/np.pi)  # End correction
        
        f = (c / (2 * np.pi)) * np.sqrt(A / (V * L_eff))
        
        return f
    
    @nb.jit(nopython=True, parallel=True)
    def apply_helmholtz_resonance(self, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                                 resonator_positions: List[Tuple[int, int, int]], resonance_freqs: List[float],
                                 current_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply Helmholtz resonance effects"""
        new_pressure = pressure.copy.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        for idx in nb.prange(len(resonator_positions)):
            i, j, k = resonator_positions[idx]
            f0 = resonance_freqs[idx]
            
            if f0 > 0:
                # Simple resonant response
                Q = 10.0  # Quality factor
                resonance_gain = Q / (1 + Q * np.abs(np.log(f0/1000)))
                
                # Apply resonance as pressure amplification at resonant frequency
                resonant_pressure = pressure[i, j, k] * resonance_gain * np.sin(2 * np.pi * f0 * current_time)
                new_pressure[i, j, k] += resonant_pressure
        
        return new_pressure, new_vx, new_vy, new_vz
    
    def update(self, layer_manager, soxel_grid, current_time: float = 0.0):
        """Apply Helmholtz resonance effects"""
        if not self.config.resonance.enable_helmholtz:
            return layer_manager
        
        # Detect resonators
        resonators = self.detect(soxel_grid)
        
        if not resonators:
            return layer_manager
        
        # Extract positions and frequencies
        positions = [r['position'] for r in resonators]
        frequencies = [self.calculate_resonance_frequency(r) for r in resonators]
        
        # Apply resonance
        new_pressure, new_vx, new_vy, new_vz = self.apply_helmholtz_resonance(
            layer_manager.pressure,
            layer_manager.velocity_x,
            layer_manager.velocity_y,
            layer_manager.velocity_z,
            positions,
            frequencies,
            current_time
        )
        
        # Update layer manager
        layer_manager.pressure = new_pressure
        layer_manager.velocity_x = new_vx
        layer_manager.velocity_y = new_vy
        layer_manager.velocity_z = new_vz
        
        return layer_manager

