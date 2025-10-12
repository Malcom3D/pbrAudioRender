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
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .fdtd_solver import FrequencyDependentFDTD, FDTDConfig
from .interface import InterfaceManager
from .resonance import ResonanceManager
from .damping import DampingManager
from .boundary_conditions import BoundaryConditions
from .termination import SimulationTermination
from ..utils.gpu_acceleration import GPUManager


@dataclass
class WavePropagationConfig:
    """Configuration for wave propagation components"""
    enable_diffraction: bool = True
    enable_reflection: bool = True
    enable_refraction: bool = True
    enable_scattering: bool = True
    enable_absorption: bool = True
    enable_resonance: bool = True
    max_interactions: int = 5
    interaction_threshold: float = 0.01


class WavePropagation:
    """
    Manages the complete wave propagation pipeline
    """
    
    def __init__(self, layer_manager, config, gpu_manager: Optional[GPUManager] = None):
        self.layer_manager = layer_manager
        self.config = config
        self.gpu = gpu_manager
        self.source_idx = layer_manager.source_idx
        
        # Initialize FDTD solver
        fdtd_config = FDTDConfig(
            courant_number=0.3,
            max_frequency=config.voxel_grid.sample_rate / 2,
            frequency_bins=64
        )
        
        self.fdtd_solver = FrequencyDependentFDTD(
            fdtd_config,
            config.voxel_grid.shape,
            config.voxel_grid.voxel_size,
            config.voxel_grid.sample_rate,
            gpu_manager
        )
        
        # Initialize interaction manager
        self.interface_manager = InterfaceManager(config, gpu_manager)
        
        # Initialize resonance manager
        self.resonance_manager = ResonanceManager(config, gpu_manager)
        
        # Initialize damping manager
        self.damping_manager = DampingManager(config, gpu_manager)
        
        # Initialize boundary conditions
        self.boundary_conditions = BoundaryConditions(config, gpu_manager)
        
        # Initialize termination checker
        self.termination = SimulationTermination(config)
        
        print(f"WavePropagation initialized for source {self.source_idx}")
    
    def update(self, soxel_grid, current_frame: int):
        """
        Perform complete wave propagation update for current frame
        """
        if self.layer_manager.ended:
            return
        
        # Get source audio sample for this frame
        source_audio_sample = self._get_source_audio_sample(soxel_grid, current_frame)
        
        # Step 1: FDTD update
        fdtd_fields = self.fdtd_solver.update_step(soxel_grid, source_audio_sample)
        
        # Step 2: Interface interactions
        interface_fields = self.interface_manager.update_step(
            fdtd_fields, soxel_grid, self.layer_manager
        )
        
        # Step 3: Resonance effects
        resonance_fields = self.resonance_manager.update_step(
            fdtd_fields, interface_fields, soxel_grid
        )
        
        # Step 4: Damping
        damped_fields = self.damping_manager.update_step(
            fdtd_fields, resonance_fields, soxel_grid
        )
        
        # Step 5: Boundary conditions
        final_fields = self.boundary_conditions.update_step(
            fdtd_fields, damped_fields, soxel_grid
        )
        
        # Step 6: Check termination
        should_terminate = self.termination.update_step(final_fields)
        
        if should_terminate:
            self.layer_manager.ended = True
            print(f"Source {self.source_idx} terminated at frame {current_frame}")
        else:
            # Update layer manager with final fields
            self.layer_manager.set_fields(final_fields)
    
    def _get_source_audio_sample(self, soxel_grid, current_frame: int) -> float:
        """Get audio sample for current frame from source"""
        # Find the source voxel for this source
        for i in range(soxel_grid.shape[0]):
            for j in range(soxel_grid.shape[1]):
                for k in range(soxel_grid.shape[2]):
                    soxel = soxel_grid.grid[i, j, k]
                    if soxel.is_source and soxel.idx == self.source_idx:
                        return soxel.audio_sample
        
        return 0.0  # No source found
    
    def get_energy(self) -> float:
        """Get total acoustic energy in this propagation layer"""
        return self.f.fdtd_solver.get_energy()
    
    def reset(self):
        """Reset all propagation components"""
        self.fdtd_solver.reset_fields()
        self.layer_manager.reset()
        self.termination.reset()

