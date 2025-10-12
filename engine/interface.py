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
Interface manager for handling interactions between different acoustic media
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .interface.absorption import AbsorptionManager
from .interface.refraction import RefractionManager
from .interface.reflection import ReflectionManager
from .interface.scattering import ScatteringManager
from .interface.diffraction import DiffractionManager

from ..utils.gpu_acceleration import GPUManager


@dataclass
class InterfaceConfig:
    """Configuration for interface interactions"""
    absorption_enabled: bool = True
    refraction_enabled: bool = True
    reflection_enabled: bool = True
    scattering_enabled: bool = True
    diffraction_enabled: bool = True
    min_impedance_ratio: float = 0.1
    max_impedance_ratio: float = 10.0


class InterfaceManager:
    """
    Manages all interface interactions between different acoustic media
    """
    
    def __init__(self, config, gpu_manager: Optional[GPUManager] = None):
        self.config = config
        self.gpu = gpu_manager
        
        # Initialize interaction managers
        self.absorption_manager = AbsorptionManager(config, gpu_manager)
        self.refraction_manager = RefractionManager(config, gpu_manager)
        self.reflection_manager = ReflectionManager(config, gpu_manager)
        self.scattering_manager = ScatteringManager(config, gpu_manager)
        self.diffraction_manager = DiffractionManager(config, gpu_manager)
    
    def update_step(self, input_fields: Dict[str, np.ndarray], 
                   soxel_grid, layer_manager) -> Dict[str, np.ndarray]:
        """
        Process all interface interactions for current fields
        """
        current_fields = input_fields.copy.copy()
        
        # Detect boundaries where material properties change
        boundaries = self._detect_boundaries(current_fields, soxel_grid)
        
        if not boundaries:
            return current_fields  # No boundaries detected
        
        # Process diffraction at boundaries
        if self.config.interface.diffraction_enabled:
            current_fields = self.diffraction_manager.apply_diffraction(
                current_fields, boundaries, soxel_grid
            )
        
        # Process absorption at boundaries
        if self.config.interface.absorption_enabled:
            current_fields = self.absorption_manager.apply_absorption(
                current_fields, boundaries, soxel_grid
            )
        
        # Process refraction through boundaries
        if self.config.interface.refraction_enabled:
            current_fields = self.refraction_manager.apply_refraction(
                current_fields, boundaries, soxel_grid
            )
        
        # Process reflection at boundaries
        if self.config.interface.reflection_enabled:
            current_fields = self.reflection_manager.apply_reflection(
                current_fields, boundaries, soxel_grid
            )
        
        # Process scattering at boundaries
        if self.config.interface.scattering_enabled:
            current_fields = self.scattering_manager.apply_scattering(
                current_fields, boundaries, soxel_grid
            )
        
        return current_fields
    
    def _detect_boundaries(self, fields: Dict[str, np.ndarray], 
                          soxel_grid) -> List[Dict[str, Any]]:
        """
        Detect boundaries between different acoustic media
        """
        boundaries = []
        shape = soxel_grid.shape
        
        # Check each voxel for material property changes with neighbors
        for i in range(1, shape[0]-1):
            for j in range(1, shape[1]-1):
                for k in range(1, shape[2]-1):
                    current_soxel = soxel_grid.grid[i, j, k]
                    
                    # Check all 6 neighbors
                    for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                        ni, nj, nk = i + di, j + dj, k + dk
                        
                        if (0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]):
                            neighbor_soxel = soxel_grid.grid[ni, nj, nk]
                            
                            # Check if there's a significant material property change
                            if self._is_boundary(current_soxel, neighbor_soxel):
                                boundary_info = {
                                    'position': (i, j, k),
                                    'neighbor_position': (ni, nj, nk),
                                    'direction': (di, dj, dk),
                                    'current_soxel': current_soxel,
                                    'neighbor_soxel': neighbor_soxel,
                                    'impedance_ratio': self._calculate_impedance_ratio(
                                        current_soxel, neighbor_soxel
                                    )
                                }
                                boundaries.append(boundary_info)
        
        return boundaries
    
    def _is_boundary(self, soxel1, soxel2, threshold: float = 0.1) -> bool:
        """Check if two soxels form a significant boundary"""
        # Check sound speed difference
        speed_diff = abs(soxel1.sound_speed - soxel2.sound_speed) / max(soxel1.sound_speed, soxel2.sound_speed)
        
        # Check density difference
        density_diff = abs(soxel1.density - soxel2.density) / max(soxel1.density, soxel2.density)
        
        # Consider it a boundary if either property differs significantly
        return speed_diff > threshold or density_diff > threshold
    
    def _calculate_impedance_ratio(self, soxel1, soxel2) -> float:
        """Calculate impedance ratio between two soxels"""
        z1 = soxel1.density * soxel1.sound_speed
        z2 = soxel2.density * soxel2.sound_speed
        
        if z2 == 0:
            return float('inf')
        
        return z1 / z2

