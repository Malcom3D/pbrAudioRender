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

import json
from typing import List, Dict, Any
from .utils.config import Config
from .utils.gpu_acceleration import GPUConfig
from .core.wave_propagation import WavePropagation
from .sources.spherical_source import SphericalSource
from .sources.plane_source import PlaneSource

class PbrAudioRender:
    def __init__(self, config_file: str):
        # Load configuration
        self.config = Config(config_file)
        
        # Setup GPU acceleration
        self.gpu_config = GPUConfig()
        
        # Initialize engine
        self.engine = WavePropagation(self.config.simulation, self.gpu_config)
        
        # Store sources and outputs
        self.sources = []
        self.outputs = []

        if self.config.sources:
            self.add_source(self.config.sources)
        elif self.config.outputs in name:
            self.add_output(self.config.outputs)
        
    def add_source(self, source_config: Dict[str, Any]):
        """Add a sound source to the simulation"""
        if source_config.type == 'spherical':
            source = SphericalSource(source_config)
        elif source_config.type == 'plane':
            source = PlaneSource(source_config)
        else:
            raise ValueError(f"Unknown source type: {source_config['type']}")
            
        self.sources.append(source)
        self.engine.add_source(source_config, source)
        
    def add_output(self, output_config: Dict[str, Any]):
        """Add an output point to the simulation"""
        from .outputs.ambisonic_output import AmbisonicOutput
        
        output = AmbisonicOutput(output_config)
        self.outputs.append(output)
        self.engine.add_output(output_config, output)
        
    def run(self):
        """Run the complete simulation"""
        print("Starting 3D acoustic simulation...")
        self.engine.run_simulation(self.sources, self.outputs)
        print("Simulation completed!")
        
    def export_ambisonic(self, output_path: str):
        """Export ambisonic results"""
        from .renderer.ambisonic_encoder import AmbisonicEncoder
        
        encoder = AmbisonicEncoder(self.config.simulation.ambisonic_order)
        for output in self.outputs:
            encoder.encode_output(output, output_path)
