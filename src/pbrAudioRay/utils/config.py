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
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

from ..lib.acoustic_shader import AcousticShader, AcousticProperties, AcousticCoefficients
from ..lib.frequency_response import SpatialFrequencyResponse

@dataclass
class SystemConfig:
    sample_rate: int = 48000
    bit_depth: int = 32
    fps: int = 24 # video fps
    fps_base: int = 1
    subframes: int = 1 # video subframes
    file_format: str = 'RAW'
    cache_path: str = "./pbrAudioCache/"
    bands_per_octave: int = 24 # frequency steps per octave
    lowest_frequency: float = 5
    higher_frequency: float = 24000.0 # Nyquist clock/2
    adr_threshold: float = None

@dataclass
class AcousticDomainConfig:
    idx: int = 0
    name: str = "acoustic_domain"
    geometry: np.ndarray = field(default_factory=lambda: np.array([]))  #vertices array
    acoustic_shader: Optional[AcousticShader] = None

@dataclass
class SourceConfig:
    idx: int
    name: str
    type: str  # "spherical", "planar"
    static: bool
    size: float = None
    width: float = None
    height: float = None
    audio_file: str = None
    pose_path: str = None
    spatial_freq_response: Optional[SpatialFrequencyResponse] = None
    spatial_freq_response_file: Optional[str] = None
    acoustic_shader: Optional[AcousticShader] = None

@dataclass
class OutputConfig:
    idx: int
    name: str
    type: str  # "AMBI", "MONO"
    static: bool
    size: float = None
    order: int = None
    spatial_arrangement_file: str = None
    pose_path: str = None
    mono_mic_type: str = None # OMNIDIRECTIONAL, CARDIOID, HYPERCARDIOID, FIGURE_8
    spatial_freq_response: Optional[SpatialFrequencyResponse] = None
    spatial_freq_response_file: Optional[str] = None
    calibration: Optional[SpatialFrequencyResponse] = None
    calibration_file: Optional[str] = None

@dataclass
class ObjectConfig:
    idx: int
    name: str
    obj_path: str
    pose_path: str
    static: bool
    stochastic_variation: bool = False
    ground: bool = False
    resonance: bool = False
    resonance_modes: int = None
    connected: Union[bool, np.ndarray] = False # for static coupled systems [[obj_idx, coupling_strength]]
    is_shard: bool = False
    fractured: Union[bool, int] = False # frame of fracture
    shard: Union[bool, np.ndarray] = False # shard of fractured object [[obj_idx, shard]]
    acoustic_shader: Optional[AcousticShader] = None

@dataclass
class WavePropagationConfig:
    max_interactions: int = 8192
    enable_interface: bool = True
    enable_resonance: bool = True
    enable_termination: bool = True
    use_dispersion_correction: bool = True # account for variations in speed due to factors like temperature and wind in the medium
    dispersion_order: int = 2
    use_extended_reaction: bool = False # from RTS
    max_modal_reaction: int = 3
    use_complex_eigenray: bool = False # infrasound
    max_complex_eigenray: int = 3

@dataclass
class FDTDConfig:
    # placeholder config class for FDTD Shaders to Render
#    enable_damping: bool = True
#    damping_coefficient: float = 0.02
#    enable_boundary: bool = True
#    boundary_type: str = "open"
#    boundary_absorption: float = 1.0
#    interaction_threshold: float = 0.01
    courant_number: float = 0.5
    max_sound_speed: float = 500.0
    stability_margin: float = 0.9

@dataclass
class InterfaceConfig:
    enable_absorption: bool = True
    enable_reflection: bool = True
    max_reflection: int = 5
    enable_scattering: bool = True
    max_scattering: int = 5
    enable_refraction: bool = True
    max_refraction: int = 5
    enable_diffraction: bool = True
    max_diffraction: int = 5
    min_impedance_ratio: float = 0.1
    max_impedance_ratio: float = 10.0

@dataclass
class ResonanceConfig:
    max_structure_resonance_modes: int = 10 # extended reaction
    decay_time_constant: float = 0.99
    resonance_threshold: float = 0.1
    enable_helmholtz: bool = True
    min_cavity_volume: float = 0.001 # cubic meters
    min_cavity_size = 8  # Cavity detection parameters: Minimum voxels for cavity
    max_cavity_size = 1000  # Cavity detection parameters: Maximum voxels for cavity
    min_neck_ratio = 0.1  # Cavity detection parameters: Minimum neck-to-cavity size ratio
    enable_parallel_wall: bool = True
    min_wall_distance: float = 0.5  # meters
    max_wall_distance: float = 20.0  # meters
    min_room_volume: float = 1.0 # cubic meters
    max_resonance_room_modes: int = 10
    enable_tube: bool = True
    min_tube_length: float = 0.3  # meters
    max_tube_length: float = 10.0  # meters
    min_tube_aspect_ratio: float = 3.0  # length/width ratio for tubes
    max_tube_cross_section: float = 1.0  # square meters

@dataclass
class TerminationConfig:
    termination_type: str = "reverberation_time"  # "sample_end", "reverberation_time", "energy_threshold", "final_frame"
    # sample_end
    samples_after: int = 100  # Samples after end of last (minimum) active sources
    min_active_sources: int = 1  # Minimum number of active sources to terminate
    # reverberation_time
    max_reverberation_time: float = 2.0
    # energy_threshold
    max_energy_threshold: float = 1e6 # Maximum energy to terminate
    min_energy_threshold: float = 1e-6 # Minimum energy to terminate

@dataclass
class AudioRecorderConfig:
    output_format: str = "npz" # npz, wav
    path: str = "./exports/audio/"

@dataclass
class AmbisonicRenderConfig:
    file_format: str = "wav" # bwf, wav
    order: int = 1
    sample_rate: int = 48000 # it's needed?
    bit_depth: int = 32 # it's needed?
    path: str = "./exports/ambisonic/"

class Config:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.data = json.load(f)
        
        self.system = SystemConfig(**self.data.get('system', {}))
        self.wave_propagation = WavePropagationConfig(**self.data.get('wave_propagation', {}))
        self.fdtd = FDTDConfig(**self.data.get('fdtd', {}))
        self.interface = InterfaceConfig(**self.data.get('interface', {}))
        self.resonance = ResonanceConfig(**self.data.get('resonance', {}))
        self.termination = TerminationConfig(**self.data.get('termination', {}))
        self.audio_recorder = AudioRecorderConfig(**self.data.get('audio_recorder', {}))
        self.ambisonic_render = AmbisonicRenderConfig(**self.data.get('ambisonic_render', {}))

        # Handle acoustic domain with nested acoustic_shader
        acoustic_domain_data = self.data.get('acoustic_domain', {})
        acoustic_shader_data = acoustic_domain_data.get('acoustic_shader', {})
        self.acoustic_domain = AcousticDomainConfig(
            **{k: v for k, v in acoustic_domain_data.items() if k != 'acoustic_shader'},
            acoustic_shader=self._create_acoustic_shader(acoustic_shader_data) if acoustic_shader_data else None
        )
        
        # Handle sources with nested acoustic_shader and spatial_freq_response
        self.sources = []
        for s in self.data.get('sources', []):
            acoustic_shader_data = s.get('acoustic_shader', {})
            spatial_freq_response_data = s.get('spatial_freq_response', {})
            
            source_config = SourceConfig(
                **{k: v for k, v in s.items() if k not in ['acoustic_shader', 'spatial_freq_response']},
                acoustic_shader=self._create_acoustic_shader(acoustic_shader_data) if acoustic_shader_data else None,
                spatial_freq_response=self._create_spatial_freq_response(spatial_freq_response_data) if spatial_freq_response_data else None
            )
            self.sources.append(source_config)
        
        # Handle outputs with nested spatial_freq_response and calibration
        self.outputs = []
        for o in self.data.get('outputs', []):
            spatial_freq_response_data = o.get('spatial_freq_response', {})
            calibration_data = o.get('calibration', {})
            
            output_config = OutputConfig(
                **{k: v for k, v in o.items() if k not in ['spatial_freq_response', 'calibration']},
                spatial_freq_response=self._create_spatial_freq_response(spatial_freq_response_data) if spatial_freq_response_data else None,
                calibration=self._create_spatial_freq_response(calibration_data) if calibration_data else None
            )
            self.outputs.append(output_config)
        
        # Handle objects with nested acoustic_shader
        self.objects = []
        for o in self.data.get('objects', []):
            acoustic_shader_data = o.get('acoustic_shader', {})
            
            object_config = ObjectConfig(
                **{k: v for k, v in o.items() if k != 'acoustic_shader'},
                acoustic_shader=self._create_acoustic_shader(acoustic_shader_data) if acoustic_shader_data else None
            )
            self.objects.append(object_config)
        
    def _create_acoustic_shader(self, shader_data: Dict[str, Any]) -> AcousticShader:
        """Create AcousticShader instance from dictionary data"""
        acoustic_props_data = shader_data.get('acoustic_properties', {})
        
        # Create AcousticCoefficients for each property
        acoustic_properties = AcousticProperties()
        
        if 'absorption' in acoustic_props_data:
            abs_data = acoustic_props_data['absorption']
            acoustic_properties.absorption = AcousticCoefficients(
                frequencies=np.array(abs_data['frequencies']),
                coefficients=np.array(abs_data['coefficients'])
            )
        
        if 'refraction' in acoustic_props_data:
            refr_data = acoustic_props_data['refraction']
            acoustic_properties.refraction = AcousticCoefficients(
                frequencies=np.array(refr_data['frequencies']),
                coefficients=np.array(refr_data['coefficients'])
            )
        
        if 'reflection' in acoustic_props_data:
            refl_data = acoustic_props_data['reflection']
            acoustic_properties.reflection = AcousticCoefficients(
                frequencies=np.array(refl_data['frequencies']),
                coefficients=np.array(refl_data['coefficients'])
            )
        
        if 'scattering' in acoustic_props_data:
            scat_data = acoustic_props_data['scattering']
            acoustic_properties.scattering = AcousticCoefficients(
                frequencies=np.array(scat_data['frequencies']),
                coefficients=np.array(scat_data['coefficients'])
            )
        
        # Create AcousticShader
        return AcousticShader(
            sound_speed=shader_data.get('sound_speed', 343.0),
            young_modulus=shader_data.get('young_modulus', []),
            poisson_ratio=shader_data.get('poisson_ratio', []),
            density=shader_data.get('density', 1.225),
            damping=shader_data.get('damping', []),
            low_frequency=shader_data.get('low_frequency', 1.0),
            high_frequency=shader_data.get('high_frequency', 24000.0),
            acoustic_properties=acoustic_properties
        )

    def _create_spatial_freq_response(self, response_data: Dict[str, Any]) -> SpatialFrequencyResponse:
        """Create SpatialFrequencyResponse instance from dictionary data"""
        return SpatialFrequencyResponse(
            azimuths=np.array(response_data.get('azimuths', [])),
            elevations=np.array(response_data.get('elevations', [])),
            frequencies=np.array(response_data.get('frequencies', [])),
            magnitude=np.array(response_data.get('magnitude', [])),
            phases=np.array(response_data.get('phases', [])) if 'phases' in response_data else None
        )

