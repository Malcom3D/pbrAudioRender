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

import trimesh
from pbrAudioCommon import np
from dataclasses import dataclass
from typing import List, Tuple

from dask import delayed, compute

from pbrAudioRay.core.entity_manager import EntityManager

from pbrAudioRay.lib.frequency_bands import FrequencyBands
from pbrAudioRay.lib.acoustic_object import AcousticObject
from pbrAudioRay.lib.geometry_data import GeometryData
from pbrAudioRay.lib.medium_properties import MediumProperties
from pbrAudioRay.lib.material_properties import MaterialProperties

from pbrAudioRay.lib.functions import _mono_to_bands

from pbrAudioRay.lib.ambisonic_ir_interpolator import AmbisonicIRInterpolator

from pbrAudioRay.sources.spherical_source import SphericalSource
from pbrAudioRay.sources.planar_source import PlanarSource

from pbrAudioRay.engine.wave_propagator import WavePropagator

from pbrAudioRay.outputs.ambisonic_output import AmbisonicOutput
from pbrAudioRay.outputs.omnidirectional_output import OmnidirectionalOutput
from pbrAudioRay.outputs.cardioid_output import CardioidOutput
from pbrAudioRay.outputs.hypercardioid_output import HypercardioidOutput
from pbrAudioRay.outputs.figure8_output import Figure8Output

# Configure Dask to use more threads
from dask import config as dask_config
#dask_config.set(scheduler='threads', num_workers=1024)
#dask_config.set(scheduler='processes', num_workers=1024)
dask_config.set(num_workers=1024)

@dataclass
class AcousticEngine:
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')

        frequency_bands = FrequencyBands(self.entity_manager)
        self.entity_manager.register('frequency_bands', frequency_bands)

        tasks = [self._add_source(source) for source in config.sources]
        tasks += [self._add_object(obj) for obj in config.objects]
        tasks += [self._add_output(output) for output in config.outputs]
        compute(*tasks)

        print('AcousticEngine: configuration loaded...')

        # Initialize geometry
        self.geometry_data: Optional[GeometryData] = None
        self.material_properties: Optional[MaterialProperties] = None
        self.medium_properties: Optional[MediumProperties] = None

        self._initialize_scene()
        self._initialize_objects()

        self.entity_manager.register('geometry_data', self.geometry_data)
        self.entity_manager.register('material_properties', self.material_properties)
        self.entity_manager.register('medium_properties', self.medium_properties)

        print('AcousticEngine: acoustic scene loaded...')

        combos = []
        for i in range(len(config.sources)):
            for j in range(len(config.outputs)):
                combos.append([config.sources[i].idx, config.outputs[j].idx])
        tasks = [self._add_solvers(combo) for combo in combos]
        compute(*tasks)

        print('AcousticEngine: wave propagator engine ready...')

    @delayed
    def _add_source(self, config):
        source_map = {
            'SPHERE': SphericalSource,
            'PLANE': PlanarSource
        }
        if 'SourceConfig' in str(type(config)) and config.type in source_map:
            source = source_map.get(config.type)(self.entity_manager, config.idx)
            self.entity_manager.register('sources', source)

    @delayed
    def _add_object(self, config):
        if 'ObjectConfig' in str(type(config)):
            obj = AcousticObject(self.entity_manager, config)
            self.entity_manager.register('objects', obj)

    @delayed
    def _add_output(self, config):
        output_map = {
            'AMBI': AmbisonicOutput,
            'OMNIDIRECTIONAL': OmnidirectionalOutput,
            'CARDIOID': CardioidOutput,
            'HYPERCARDIOID': HypercardioidOutput,
            'FIGURE_8': Figure8Output
        }
        if 'OutputConfig' in str(type(config)):
            if config.type == 'AMBI':
                config_type = config.type
            elif config.type == 'MONO' and config.microphone_type in output_map:
                config_type = config.microphone_type
            output = output_map.get(config_type)(self.entity_manager, config.idx)
            self.entity_manager.register('outputs', output)

    @delayed
    def _add_solvers(self, combo):
        wave_propagator = WavePropagator(self.entity_manager, combo)
        self.entity_manager.register('wave_propagators', wave_propagator)

    def compute(self):
        config = self.entity_manager.get('config')

        start_frame = config.system.start_frame
        end_frame = config.system.end_frame

        for frame_idx in range(end_frame - start_frame):
            self._compute_frame(frame_idx)

        self.render()

    def render(self):
        config = self.entity_manager.get('config')
        # interpolate x bands_idx IRs for wave_propagators[index].combo
        combos = []
        for i in range(len(config.sources)):
            for j in range(len(config.outputs)):
                combos.append([config.sources[i].idx, config.outputs[j].idx])
        ir_tasks = [self._render_audio(combo) for combo in combos]
        ir_results = compute(*ir_tasks)

    @delayed
    def _render_audio(self, combo):
        interpolator = AmbisonicIRInterpolator(self.entity_manager, combo)

        # run acoustic_render.compute to convolve x source wave file with interpolated x bands_idx IRs
        convolved_audio = interpolator.smooth_convolve()

        # save convolved audio
        interpolator.save_output()
        
    def _compute_frame(self, frame_idx: int):
        wave_propagators = self.entity_manager.get('wave_propagators')
        tasks = [wave_propagators[index].compute(frame_idx) for index in wave_propagators.keys()]
        results = compute(*tasks)

    def _initialize_scene(self):
        """Initialize the acoustic domain scene."""
        config = self.entity_manager.get('config')
        frequency_bands = self.entity_manager.get('frequency_bands')
        n_bands = len(frequency_bands.get_bands())

        ac_geometry = np.array(config.acoustic_domain.geometry)
        ac_max = np.max(ac_geometry, axis=0)
        ac_min = np.min(ac_geometry, axis=0)

        mesh = trimesh.creation.box(bounds=(ac_min, ac_max))
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32)

        mesh_info = mesh.vertices[mesh.faces]
        scene_info = np.full((mesh_info.shape[0],), [-1], dtype=np.int32)

        # Initialize medium properties
        sound_speed = config.acoustic_domain.acoustic_shader.sound_speed
        density = config.acoustic_domain.acoustic_shader.density
        temperature = config.acoustic_domain.acoustic_shader.temperature
        impedance = config.acoustic_domain.acoustic_shader.impedence

        alpha, beta = self._compute_acoustic_domain_coefficients(sound_speed, density, temperature, impedance)

        self.medium_properties = MediumProperties(
            speed=sound_speed,
            alpha=alpha,
            beta=beta,
            density=density,
            temperature=temperature,
            impedance=impedance
        )

        # Initialize acoustic properties for domain
        n_faces = vertices[faces].shape[0]
        self.material_properties = MaterialProperties(
            absorption_coeffs=np.full((n_faces, n_bands), 1.0, dtype=np.float32),
            absorption_phases=np.full((n_faces, n_bands), 0.0, dtype=np.float32),
            reflection_coeffs=np.full((n_faces, n_bands), 0.0, dtype=np.float32),
            reflection_phases=np.full((n_faces, n_bands), 0.0, dtype=np.float32),
            transmission_coeffs=np.full((n_faces, n_bands), 0.0, dtype=np.float32),
            transmission_phases=np.full((n_faces, n_bands), 0.0, dtype=np.float32),
            scattering_coeffs=np.full((n_faces, n_bands), 0.0, dtype=np.float32),
            scattering_phases=np.full((n_faces, n_bands), 0.0, dtype=np.float32),
            roughness=np.full((n_faces, 1), 0.0, dtype=np.float32)
        )

        self.geometry_data = GeometryData(
            vertices=vertices,
            faces=faces,
            mesh_info=mesh_info,
            scene_info=scene_info
        )

    def _initialize_objects(self):
        """Initialize scene objects."""
        config = self.entity_manager.get('config')
        frequency_bands = self.entity_manager.get('frequency_bands')
        objects = self.entity_manager.get('objects')
        n_bands = len(frequency_bands.get_bands())

        for key in objects:
            obj_config = objects[key].config_obj
            vertices, vertex_normals, faces = objects[key].get_data(-1)

            self.geometry_data.mesh_info = np.append(self.geometry_data.mesh_info, vertices[faces], axis=0)
            self.geometry_data.scene_info = np.append(self.geometry_data.scene_info, np.full((vertices[faces].shape[0],), [obj_config.idx], dtype=np.int32))

            # Get object acoustic properties
            n_faces = vertices[faces].shape[0]
            roughness = obj_config.acoustic_shader.roughness
            self.material_properties.roughness = np.append(self.material_properties.roughness, np.full((n_faces, 1), [roughness], dtype=np.float32), axis=0)

            for prop_name in ['absorption', 'reflection', 'transmission', 'scattering']:
                if hasattr(obj_config.acoustic_shader.acoustic_properties, prop_name):
                    prop = getattr(obj_config.acoustic_shader.acoustic_properties, prop_name)
                    coeffs, phases = prop.get_bands_avg(frequency_bands.get_bands())

                    existing_coeffs = getattr(self.material_properties, f'{prop_name}_coeffs')
                    existing_phases = getattr(self.material_properties, f'{prop_name}_phases')

                    setattr(self.material_properties, f'{prop_name}_coeffs', np.append(existing_coeffs, np.full((n_faces, n_bands), coeffs, dtype=np.float32), axis=0))
                    setattr(self.material_properties, f'{prop_name}_phases', np.append(existing_phases, np.full((n_faces, n_bands), phases, dtype=np.float32), axis=0))

    def _compute_acoustic_domain_coefficients(self, c: float, rho: float, T: float, Z: float):
        """Compute absorption and phase shift coefficients for air."""
        frequency_bands = self.entity_manager.get('frequency_bands')
        freqs = np.unique(frequency_bands.get_bands())[:-1]

        T_K = T + 273.15
        omega = 2 * np.pi * freqs

        # Constants for air
        mu = 1.846e-5  # Dynamic viscosity (Pa·s)
        kappa = 0.0262  # Thermal conductivity (W/m·K)
        Cp = 1005  # Specific heat at constant pressure (J/kg·K)
        Cv = 718  # Specific heat at constant volume (J/kg·K)
        gamma_specific = Cp / Cv

        # Viscous and thermal contributions
        alpha_visc = (omega**2 * mu) / (2 * rho * c**3)
        alpha_therm = (omega**2 * kappa * (gamma_specific - 1)) / (2 * rho * c**3 * Cp)

        alpha = alpha_visc + alpha_therm
        beta = omega / c

        return alpha, beta
