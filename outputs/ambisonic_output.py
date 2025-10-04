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
import json
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
from .omnidirectional_output import OmnidirectionalOutput
from .figure8_output import Figure8Output
from .cardioid_output import CardioidOutput

@dataclass
class AmbisonicOutput:
    config: dict
    output_points: List = None
    spatial_arrangement: Dict = None
    
    def __post_init__(self):
        """Load spatial arrangement and create output points"""
        self.load_spatial_arrangement()
        self.create_output_points()
        
    def load_spatial_arrangement(self):
        """Load spatial arrangement from JSON file"""
        arrangement_file = self.config.get('spatial_arrangement')
        if arrangement_file and Path(arrangement_file).exists():
            with open(arrangement_file, 'r') as f:
                self.spatial_arrangement = json.load(f)
        else:
            # Default: tetrahedral arrangement
            self.spatial_arrangement = {
                "name": "4ch_tetrahedral",
                "outputs": [
                    {"id": 0, "type": "omnidirectional", "position": [1, 0, 0]},
                    {"id": 1, "type": "omnidirectional", "position": [-1/3, 2*np.sqrt(2)/3, 0]},
                    {"id": 2, "type": "omnidirectional", "position": [-1/3, -np.sqrt(2)/3, np.sqrt(6)/3]},
                    {"id": 3, "type": "omnidirectional", "position": [-1/3, -np.sqrt(2)/3, -np.sqrt(6)/3]}
                ]
            }
    
    def create_output_points(self):
        """Create output point objects based on spatial arrangement"""
        self.output_points = []
        
        for output_config in self.spatial_arrangement['outputs']:
            # Calculate absolute position
            rel_pos = np.array(output_config['position'])
            abs_pos = tuple(np.array(self.position) + rel_pos)
            
            # Create output config
            config = {
                'position': abs_pos,
                'type': output_config['type'],
                'rotation': output_config.get('rotation', (0.0, 0.0, 0.0))
            }
            
            # Create appropriate output type
            if output_config['type'] == 'omnidirectional':
                output = OmnidirectionalOutput(config)
            elif output_config['type'] == 'figure8':
                output = Figure8Output(config)
            elif output_config['type'] == 'cardioid':
                output = CardioidOutput(config)
            else:
                raise ValueError(f"Unknown output type: {output_config['type']}")
                
            self.output_points.append(output)
    
    def capture_frame(self, soxels, frame: int) -> np.ndarray:
        """Capture audio from all output points for current frame"""
        samples = []
        
        for output in self.output_points:
            x, y, z = output.position
            
            # Ensure position is within grid bounds
            if (0 <= x < soxels.shape[0] and 
                0 <= y < soxels.shape[1] and 
                0 <= z < soxels.shape[2]):
                
                soxel = soxels[x, y, z]
                pressure = soxel.pressure
                velocity = soxel.velocity
                
                # Apply microphone directivity
                sample = output.capture_pressure(pressure, velocity, output.position)
                samples.append(sample)
            else:
                samples.append(0.0)
        
        return np.array(samples)
    
    @property
    def position(self):
        return self.config['position']
