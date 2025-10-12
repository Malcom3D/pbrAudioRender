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
3D Acoustic Wave Propagation Simulation Engine
"""

from .core.acoustic_engine import AcousticEngine
from .core.soxel import Soxel, SoxelGrid
from .core.zarr_store import ZarrStore
from .utils.config import Config
from .renderer.ambisonic_encoder import AmbisonicEncoder

__version__ = "0.1.0"
__author__ = "Acoustic Simulation Engine"
__description__ = "3D acoustic wave propagation simulation with voxel-based rendering"

class pbrAudioRender:
    """Main class for running 3D acoustic simulations"""
    
    def __init__(self, config_file: str):
        # Load configuration
        self.config = Config(config_file)
        
        # Initialize core components
        self.soxel_grid = SoxelGrid(
            self.config.voxel_grid,
            self.config.sources,
            self.config.objects
        )
        
        self.engine = AcousticEngine(
            self.config,
            self.soxel_grid
        )
        
        # Initialize renderers
        self.ambisonic_encoder = AmbisonicEncoder(
            self.config.voxel_grid.sample_rate,
            self.config.voxel_grid.bit_depth,
            self.config.outputs
        )
        
    def run_simulation(self):
        """Run the main simulation loop"""
        self.engine.run_simulation()
        
    def encode_ambisonic(self):
        """Encode simulation results to Ambisonic B-format"""
        self.ambisonic_encoder.encode()

    def export_results(self):
        """Export all simulation results"""
        self.encode_ambisonic()
        # Add other export methods here

__all__ = ['pbrAudioRender', 'Config', 'AcousticEngine', 'SoxelGrid']

