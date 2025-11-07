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

import numba as nb
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from ...lib.base import Configurable, GPUEnabled


class TubeResonance(Configurable, GPUEnabled):
    """Handle tube resonator effects (open-open, open-closed, closed-closed)"""
    
    def __init__(self, config=None, gpu_manager=None):
        super().__init__(config)
        GPUEnabled.__init__(self, gpu_manager)
        
        self.min_tube_length = getattr(config.resonance, 'min_tube_length', 0.1)
        self.max_resonance_modes = getattr(config.resonance, 'max_resonance_modes', 10)
    
    def detect(self, soxel_grid) -> List[Dict]:
        """Detect tube-like structures in the scene"""
        tubes = []
        
        # Look for elongated structures in all three axes
        for axis in [0, 1, 2]:  # x, y, z axes
            axis_tubes = self._find_tubes_on_axis(soxel_grid, axis)
            tubes.extend(axis_tubes)
        
        return tubes
    
    def _find_tubes_on_axis(self, soxel_grid, axis: int) -> List[Dict]:
        """Find tube-like structures along given axis"""
        tubes = []
        
        if axis == 0:  # x-axis tubes
            for j in range(soxel_grid.shape[1]):
                for k in range(soxel_grid.shape[2]):
                    tube = self._find_tube_in_column(soxel_grid, axis, j, k)
                    if tube:
                        tubes.append(tube)
        
        elif axis == 1:  # y-axis tubes
            for i in range(soxel_grid.shape[0]):
                for k in range(soxel_grid.shape[2]):
                    tube = self._find_tube_in_column(soxel_grid, axis, i, k)
                    if tube:
                        tubes.append(tube)
        
        else:  # z-axis tubes
            for i in range(soxel_grid.shape[0]):
                for j in range(soxel_grid.shape[1]):
                    tube = self._find_tube_in_column(soxel_grid, axis, i, j)
                    if tube:
                        tubes.append(tube)
        
        return tubes
    
    def _find_tube_in_column(self, soxel_grid, axis: int, coord1: int, coord2: int) -> Optional[Dict]:
        """Find tube in a column along given axis"""
        if axis == 0:
            column = soxel_grid.soxel_types[:, coord1, coord2]
        elif axis == 1:
            column = soxel_grid.soxel_types[coord1, :, coord2]
        else:
            column = soxel_grid.soxel_types[coord1, coord2, :]
        
        # Find continuous non-object segments (potential tube interiors)
        air_segments = []
        in_segment = False
        start_idx = 0
        
        for idx, voxel_type in enumerate(column):
            if voxel_type != 2 and not in_segment:  # Start of air segment
                in_segment = True
                start_idx = idx
            elif voxel_type == 2 and in_segment:  # End of air segment
                in_segment = False
                air_segments.append((start_idx, idx-1))
        
        if in_segment:
            air_segments.append((start_idx, len(column)-1))
        
        # Find tube-like segments (long narrow air passages)
        for start, end in air_segments:
            length = end - start + 1
            
            if length >= 3:  # Minimum tube length
                # Check if surrounded by objects (forming tube walls)
                if self._is_tube_structure(soxel_grid, axis, coord1, coord2, start, end):
                    tube = {
                        'axis': axis,
                        'coord1': coord1,
                        'coord2': coord2,
                        'start': start,
                        'end': end,
                        'length': length * soxel_grid.voxel_size,
                        'type': self._determine_tube_type(soxel_grid, axis, coord1, coord2, start, end)
                    }
                    tubes.append(tube)
        
        return tubes if tubes else None
    
    def _is_tube_structure(self, soxel_grid, axis: int, coord1: int, coord2: int, 
                          start: int, end: int) -> bool:
        """Check if air segment forms a tube structure"""
        # Check if surrounded by objects in perpendicular directions
        if axis == 0:  # x-axis tube
            # Check y and z directions for enclosing walls
            for i in range(start, end + 1):
                # Check adjacent positions in y and z
                if (coord1 > 0 and soxel_grid.soxel_types[i, coord1-1, coord2] != 2) or \
                   (coord1 < soxel_grid.shape[1]-1 and soxel_grid.soxel_types[i, coord1+1, coord2] != 2) or \
                   (coord2 > 0 and soxel_grid.soxel_types[i, coord1, coord2-1] != 2) or \
                   (coord2 < soxel_grid.shape[2]-1 and soxel_grid.soxel_types[i, coord1, coord2+1] != 2):
                    return False
        # Similar checks for other axes...
        
        return True
    
    def _determine_tube_type(self, soxel_grid, axis: int, coord1: int, coord2: int, 
                            start: int, end: int) -> str:
        """Determine tube type (open-open, open-closed, closed-closed)"""
        # Check endpoints
        start_open = self._is_tube_end_open(soxel_grid, axis, coord1, coord2, start, -1)
        end_open = self._is_tube_end_open(soxel_grid, axis, coord1, coord2, end, 1)
        
        if start_open and end_open:
            return "open_open"
        elif start_open or end_open:
            return "open_closed"
        else:
            return "closed_closed"
    
    def _is_tube_end_open(self, soxel_grid, axis: int, coord1: int, coord2: int, 
                         position: int, direction: int) -> bool:
        """Check if tube end is open to larger space"""
        # Simplified check - in practice, use more sophisticated cavity detection
        if axis == 0:
            check_pos = position + direction
            if 0 <= check_pos < soxel_grid.shape[0]:
                return soxel_grid.soxel_types[check_pos, coord1, coord2] != 2
        # Similar for other axes...
        
        return False
    
    def calculate_resonance_frequency(self, tube: Dict) -> List[float]:
        """Calculate tube resonance frequencies"""
        tube_type = tube['type']
        length = tube['length']
        c = 343.0  # sound speed
        
        frequencies = []
        
        if tube_type == "open_open":
            # f_n = (n * c) / (2 * L) for n = 1, 2, 3, ...
            for n in range(1, self.max_resonance_modes + 1):
                f = (n * c) / (2 * length)
                if f <= 20000:  # Audible range
                    frequencies.append(f)
        
        elif tube_type == "open_closed":
            # f_n = ((2n - 1) * c) / (4 * L) for n = 1, 2, 3, ...
            for n in range(1, self.max_resonance_modes + 1):
                f = ((2 * n - 1) * c) / (4 * length)
                if f <= 20000:
                    frequencies.append(f)
        
        else:  # closed_closed
            # Same as open_open but with different boundary conditions
            for n in range(1, self.max_resonance_modes + 1):
                f = (n * c) / (2 * length)
                if f <= 20000:
                    frequencies.append(f)
        
        return frequencies
    
    @nb.jit(nopython=True, parallel=True)
    def apply_tube_resonance(self, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                            tubes: List[Dict], resonance_freqs: List[List[float]],
                            current_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply tube resonance effects"""
        new_pressure = pressure.copy()
        new_vx = vx.copy()
        new_vy = vy.copy()
        new_vz = vz.copy()
        
        for tube_idx in nb.prange(len(tubes)):
            tube = tubes[tube_idx]
            freqs = resonance_freqs[tube_idx]
            
            axis = tube['axis']
            start, end = tube['start'], tube['end']
            coord1, coord2 = tube['coord1'], tube['coord2']
            
            for f in freqs:
                if f > 0:
                    # Create standing wave pattern in tube
                    for pos in range(start, end + 1):
                        if axis == 0:
                            # Calculate position along tube (normalized)
                            x_norm = (pos - start) / (end - start)
                            
                            # Standing wave pattern based on tube type
                            if tube['type'] == "open_open":
                                wave_shape = np.sin(np.pi * x_norm)
                            elif tube['type'] == "open_closed":
                                wave_shape = np.sin(np.pi * x_norm / 2)
                            else:  # closed_closed
                                wave_shape = np.cos(np.pi * x_norm)
                            
                            # Apply resonance
                            resonance_gain = 5.0  # Quality factor
                            resonant_pressure = pressure[pos, coord1, coord2] * wave_shape * resonance_gain
                            new_pressure[pos, coord1, coord2] += resonant_pressure
        
        return new_pressure, new_vx, new_vy, new_vz
    
    def update(self, layer_manager, soxel_grid, current_time: float = 0.0):
        """Apply tube resonance effects"""
        if not self.config.resonance.enable_tube:
            return layer_manager
        
        # Detect tubes
        tubes = self.detect(soxel_grid)
        
        if not tubes:
            return layer_manager
        
        # Calculate resonance frequencies
        resonance_freqs = [self.calculate_resonance_frequency(tube) for tube in tubes]
        
        # Apply resonance
        new_pressure, new_vx, new_vy, new_vz = self.apply_tube_resonance(
            layer_manager.pressure,
            layer_manager.velocity_x,
            layer_manager.velocity_y,
            layer_manager.velocity_z,
            tubes,
            resonance_freqs,
            current_time
        )
        
        # Update layer manager
        layer_manager.pressure = new_pressure
        layer_manager.velocity_x = new_vx
        layer_manager.velocity_y = new_vy
        layer_manager.velocity_z = new_vz
        
        return layer_manager

