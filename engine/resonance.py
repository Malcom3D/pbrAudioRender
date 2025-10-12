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
Resonance manager for detecting and handling acoustic resonances in the simulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .resonance.helmholtz import HelmholtzResonator
from .resonance.parallel_wall import ParallelWallResonator
from .resonance.tube import TubeResonator
from ..utils.gpu_acceleration import GPUManager


@dataclass
class ResonanceConfig:
    """Configuration for resonance detection and handling"""
    enable_helmholtz: bool = True
    enable_parallel_wall: bool = True
    enable_tube: bool = True
    min_cavity_volume: float = 0.001  # m³
    max_resonance_modes: int = 10
    resonance_threshold: float = 0.1
    decay_time_constant: float = 0.99


class ResonanceManager:
    """
    Manages detection and simulation of acoustic resonances.
    Handles Helmholtz resonators, tube resonances, and parallel wall resonances.
    """
    
    def __init__(self, config, gpu_manager: Optional[GPUManager] = None):
        self.config = config.resonance
        self.gpu = gpu_manager
        
        # Initialize resonance detectors
        self.helmholtz_detector = HelmholtzResonator(config, gpu_manager)
        self.parallel_wall_detector = ParallelWallResonator(config, gpu_manager)
        self.tube_detector = TubeResonator(config, gpu_manager)
        
        # Resonance cache
        self.detected_resonances = []
        self.resonance_modes = {}
        
        print("ResonanceManager initialized")
    
    def update_step(self, fdtd_fields: Dict[str, np.ndarray],
                   interface_fields: Dict[str, np.ndarray],
                   soxel_grid) -> Dict[str, np.ndarray]:
        """
        Apply resonance effects to acoustic fields.
        
        Args:
            fdtd_fields: Fields from FDTD solver
            interface_fields: Fields after interface interactions
            soxel_grid: Current SoxelGrid state
        
        Returns:
            Fields with resonance effects applied
        """
        result_fields = interface_fields.copy()
        
        # Detect resonances in the current scene
        self._detect_resonances(soxel_grid)
        
        # Apply resonance effects
        if self.config.enable_helmholtz:
            result_fields = self.helmholtz_detector.apply_resonance(
                result_fields, self.detected_resonances, soxel_grid
            )
        
        if self.config.enable_tube:
            result_fields = self.tube_detector.apply_resonance(
                result_fields, self.detected_resonances, soxel_grid
            )
        
        if self.config.enable_parallel_wall:
            result_fields = self.parallel_wall_detector.apply_resonance(
                result_fields, self.detected_resonances, soxel_grid
            )
        
        return result_fields
    
    def _detect_resonances(self, soxel_grid):
        """Detect potential resonances in the current scene"""
        self.detected_resonances = []
        
        # Detect Helmholtz resonators
        if self.config.enable_helmholtz:
            helmholtz_resonances = self.helmholtz_detector.detect(soxel_grid)
            self.detected_resonances.extend(helmholtz_resonances)
        
        # Detect tube resonances
        if self.config.enable_tube:
            tube_resonances = self.tube_detector.detect(soxel_grid)
            self.detected_resonances.extend(tube_resonances)
        
        # Detect parallel wall resonances
        if self.config.enable_parallel_wall:
            wall_resonances = self.parallel_wall_detector.detect(soxel_grid)
            self.detected_resonances.extend(wall_resonances)
    
    def calculate_resonance_frequencies(self, soxel_grid, 
                                      cavity_voxels: List[Tuple[int, int, int]]) -> List[float]:
        """Calculate resonance frequencies for a cavity"""
        if not cavity_voxels:
            return []
        
        # Calculate cavity dimensions
        positions = np.array(cavity_voxels)
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)
        dimensions = (max_pos - min_pos) * soxel_grid.voxel_size
        
        # Get average sound speed in cavity
        sound_speeds = []
        for i, j, k in cavity_voxels:
            soxel = soxel_grid.grid[i, j, k]
            sound_speeds.append(soxel.sound_speed)
        
        avg_sound_speed = np.mean(sound_speeds)
        
        # Calculate fundamental resonance frequencies for 3D cavity
        frequencies = []
        for nx, ny, nz in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            if dimensions[0] > 0 and nx > 0:
                freq = (avg_sound_speed / 2) * np.sqrt(
                    (nx / dimensions[0])**2 + 
                    (ny / dimensions[1])**2 + 
                    (nz / dimensions[2])**2
                )
                frequencies.append(freq)
        
        return sorted(frequencies)
    
    def apply_resonance_mode(self, fields: Dict[str, np.ndarray],
                           resonance_mode: Dict[str, Any],
                           soxel_grid) -> Dict[str, np.ndarray]:
        """Apply a specific resonance mode to the fields"""
        result_fields = fields.copy()
        pressure = result_fields['pressure'].copy()
        
        # Get resonance parameters
        frequency = resonance_mode['frequency']
        strength = resonance_mode['strength']
        cavity_voxels = resonance_mode['cavity_voxels']
        
        # Calculate resonance excitation
        excitation = self._calculate_resonance_excitation(
            fields, cavity_voxels, frequency, soxxel_grid
        )
        
        # Apply resonance to pressure field
        for i, j, k in cavity_voxels:
            # Resonance adds to existing pressure
            pressure[i, j, k] += excitation * strength
        
        result_fields['pressure'] = pressure
        return result_fields
    
    def _calculate_resonance_excitation(self, fields: Dict[str, np.ndarray],
                                      cavity_voxels: List[Tuple[int, int, int]],
                                      frequency: float,
                                      soxel_grid) -> float:
        """Calculate how much a cavity is excited at a resonance frequency"""
        if not cavity_voxels:
            return 0.0
        
        # Calculate average pressure in cavity
        cavity_pressure = 0.0
        for i, j, k in cavity_voxels:
            cavity_pressure += fields['pressure'][i, j, k]
        
        cavity_pressure /= len(cavity_voxels)
        
        # Simple resonance model - in practice, use more sophisticated models
        # based on the cavity geometry and boundary conditions
        excitation = cavity_pressure * np.sin(2 * np.pi * frequency * soxel_grid.current_time)
        
        return excitation
    
    def get_resonance_energy(self) -> float:
        """Get total energy in resonance modes"""
        total_energy = 0.0
        for resonance in self.detected_resonances:
            total_energy += resonance.get('energy', 0.0)
        return total_energy

