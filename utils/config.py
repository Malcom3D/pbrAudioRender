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
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class SystemConfig:
    use_gpu: bool = True
    compute_device: str = "cuda"
    device_id: int = 0
    num_streams: int = 4
    memory_limit: Optional[int] = None
    export_vdb: bool = False
    vdb_export_path: str = "./exports/vdb/"
    max_workers: int = 4

@dataclass
class VoxelGridConfig:
    name: str = "acoustic_domain"
    sample_frame_limit: Optional[int] = None
    max_reflections: int = 5
    max_resonances: int = 3
    max_reverberation_time: float = 2.0
    sample_rate: int = 48000
    bit_depth: int = 32
    ambisonic_order: int = 1
    ambisonic_path: str = "./exports/ambisonic/"
    shape: tuple = (64, 64, 64)
    voxel_size: float = 0.1
    default_sound_speed: float = 343.0
    default_density: float = 1.2
    export_vdb: bool = False
    vdb_export_path: str = "./exports/vdb/"

@dataclass
class SourceConfig:
    idx: int
    name: str
    type: str  # "spherical", "plane"
    geometry: Dict[str, Any]
    audio_file: str
    position_file: str
    frequency_response: Optional[Dict] = None
    frequency_response_file: Optional[str] = None
    directivity_pattern: Optional[Dict] = None
    directivity_pattern_file: Optional[str] = None
    acoustic_shader: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OutputConfig:
    idx: int
    name: str
    type: str  # "omnidirectional", "cardioid", "figure8"
    spatial_arrangement_file: str
    render_output_path: str
    position_file: str
    frequency_response: Optional[Dict] = None
    frequency_response_file: Optional[str] = None
    directivity_pattern: Optional[Dict] = None
    directivity_pattern_file: Optional[str] = None

@dataclass
class ObjectConfig:
    name: str
    acoustic_shader: Dict[str, Any]
    obj_files: List[str]  # per-frame OBJ files

class Config:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.data = json.load(f)
        
        self.system = SystemConfig(**self.data.get('system', {}))
        self.voxel_grid = VoxelGridConfig(**self.data.get('acoustic_domain', {}))
        self.sources = [SourceConfig(**s) for s in self.data.get('sources', [])]
        self.outputs = [OutputConfig(**o) for o in self.data.get('outputs', [])]
        self.objects = [ObjectConfig(**o) for o in self.data.get('objects', [])]

