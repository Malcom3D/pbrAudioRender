import time
import numpy as np
import trimesh
import os
import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import zarr
import zarrs

import resampy
import soundfile as sf
from embreex import rtcore_scene as rtcs
from embreex.mesh_construction import TriangleMesh
from scipy.signal import fftconvolve, convolve

from pbrAudioRay.core.entity_manager import EntityManager
from pbrAudioRay.core.acoustic_engine import AcousticEngine
from pbrAudioRay.lib.frequency_bands import FrequencyBands
from pbrAudioRay.lib.functions import _compute_rayleigh_damping, _mono_to_bands

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

@dataclass
class GeometryData:
    """Holds geometry and scene information."""
    vertices: np.ndarray
    faces: np.ndarray
    mesh_info: np.ndarray
    scene_info: np.ndarray
    normals: Optional[np.ndarray] = None


@dataclass
class MaterialProperties:
    """Holds acoustic properties for materials."""
    absorption_coeffs: np.ndarray
    absorption_phases: np.ndarray
    reflection_coeffs: np.ndarray
    reflection_phases: np.ndarray
    refraction_coeffs: np.ndarray
    refraction_phases: np.ndarray
    scattering_coeffs: np.ndarray
    scattering_phases: np.ndarray
    roughness: np.ndarray

@dataclass
class MediumProperties:
    """Holds medium (air/fluid) properties."""
    speed: float
    alpha: np.ndarray
    beta: np.ndarray
    density: float = 1.225
    temperature: float = 20.0
    impedance: float = 413.3

@dataclass
class ZarrRayData:
    """Holds ray tracing data using Zarr arrays for memory efficiency."""
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.store_path = os.path.join(config.system.cache_path, "rays.zarr")
        self._current_size = 0

        # Remove existing store if it exists
        if os.path.exists(self.store_path):
            import shutil
            shutil.rmtree(self.store_path)

        # Initialize Zarr store
        self.compressors = zarr.codecs.BloscCodec(cname='blosclz', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
        self.store = zarr.create_group(store=self.store_path)

        # Initialize empty arrays (will be resized as needed)
        self.origins = self._create_array('origins', shape=(0, 3), dtype=np.float32)
        self.origins_idx = self._create_array('origins_idx', shape=(0, 1), dtype=np.int32)
        self.origins_bands = self._create_array('origins_bands', shape=(0, 1), dtype=np.int32)
        self.destinations = self._create_array('destinations', shape=(0, 3), dtype=np.float32)
        self.destinations_idx = self._create_array('destinations_idx', shape=(0, 1), dtype=np.float32)
        self.directions = self._create_array('directions', shape=(0, 3), dtype=np.float32)
        self.energies = self._create_array('energies', shape=(0, 1), dtype=np.float32)
        self.phases = self._create_array('phases', shape=(0, 1), dtype=np.float32)
        self.delay = self._create_array('delay', shape=(0, 1), dtype=np.float32)

    def _create_array(self, name: str, shape: tuple, dtype: np.dtype) -> zarr.Array:
        """Create a Zarr array with chunking."""
        chunk_size = calculate_chunk_size(shape, dtype)
        return self.store.empty(name=name, shape=shape, chunks=(chunk_size), dtype=dtype)
        
    @staticmethod
    def calculate_chunk_size(shape, dtype):
        """Calculate optimal chunk size for Zarr arrays"""
        data_size_bytes = np.dtype(dtype).itemsize
        target_chunk_bytes = 1024 * 1024  # 1MB
        chunk_dim = int((target_chunk_bytes / data_size_bytes) ** (1/shape[1]))
        return tuple(chunk_dim for _ in range(shape[1]))

    def apply_mask(self, mask: np.array):
        # compute zarr new size
        new_size = int(np.count_nonzero(mask))

        # Filter all ray data to temp array
        temp_origins = self.origins.get_orthogonal_selection(mask)
        temp_origins_idx = self.origins_idx[:][mask]
        temp_origins_bands = self.origins_bands[:][mask]
        temp_destinations = self.destinations.get_orthogonal_selection(mask)
        temp_destinations_idx = self.destinations_idx[:][mask]
        temp_directions = self.directions.get_orthogonal_selection(mask)
        temp_energies = self.energies[:][mask]
        temp_phases = self.phases[:][mask]
        temp_delay = self.delay[:][mask]

        # resize zarr array
        self.origins.resize((new_size, self.origins.shape[1]))
        self.origins_idx.resize((new_size, self.origins_idx.shape[1]))
        self.origins_bands.resize((new_size, self.origins_bands.shape[1]))
        self.destinations.resize((new_size, self.destinations.shape[1]))
        self.destinations_idx.resize((new_size, self.destinations_idx.shape[1]))
        self.directions.resize((new_size, self.directions.shape[1]))
        self.energies.resize((new_size, self.energies.shape[1]))
        self.phases.resize((new_size, self.phases.shape[1]))
        self.delay.resize((new_size, self.delay.shape[1]))

        # copy temp array to zarr
        self.origins.set_mask_selection(np.ones_like(self.origins, dtype=bool), temp_origins.flatten())
        self.origins_idx.set_mask_selection(np.ones_like(self.origins_idx, dtype=bool), temp_origins_idx.flatten())
        self.origins_bands.set_mask_selection(np.ones_like(self.origins_bands, dtype=bool), temp_origins_bands.flatten())
        self.destinations.set_mask_selection(np.ones_like(self.destinations, dtype=bool), temp_destinations.flatten())
        self.destinations_idx.set_mask_selection(np.ones_like(self.destinations_idx, dtype=bool), temp_destinations_idx.flatten())
        self.directions.set_mask_selection(np.ones_like(self.directions, dtype=bool), temp_directions.flatten())
        self.energies.set_mask_selection(np.ones_like(self.energies, dtype=bool), temp_energies.flatten())
        self.phases.set_mask_selection(np.ones_like(self.phases, dtype=bool), temp_phases.flatten())
        self.delay.set_mask_selection(np.ones_like(self.delay, dtype=bool), temp_delay.flatten())

        # delete temp array
        del temp_origins
        del temp_origins_idx
        del temp_origins_bands
        del temp_destinations
        del temp_destinations_idx
        del temp_directions
        del temp_energies
        del temp_phases
        del temp_delay

@dataclass
class ZarrOutputData:
    """Holds accumulated output data using Zarr arrays."""
    entity_manager: EntityManager
    chunk_size: int = 32

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.store_path = os.path.join(config.system.cache_path, "output_data.zarr")
        self._current_size = 0

        # Remove existing store if it exists
        if os.path.exists(self.store_path):
            import shutil
            shutil.rmtree(self.store_path)

        # Initialize Zarr store
        self.compressors = zarr.codecs.BloscCodec(cname='blosclz', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
        self.store = zarr.create_group(store=self.store_path)

        # Initialize arrays
        self.source = self._create_array('source', shape=(0, 1), dtype=np.float32)
        self.bands = self._create_array('bands', shape=(0, 1), dtype=np.float32)
        self.energies = self._create_array('energies', shape=(0, 1), dtype=np.float32)
        self.phases = self._create_array('phases', shape=(0, 1), dtype=np.float32)
        self.delay = self._create_array('delay', shape=(0, 1), dtype=np.float32)
        self.origins = self._create_array('origins', shape=(0, 3), dtype=np.float32)
        self.directions = self._create_array('directions', shape=(0, 3), dtype=np.float32)
        self.destinations = self._create_array('destinations', shape=(0, 3), dtype=np.float32)
        self.destinations_idx = self._create_array('destinations_idx', shape=(0, 1), dtype=np.float32)

    def _create_array(self, name: str, shape: tuple, dtype: np.dtype) -> zarr.Array:
        """Create a Zarr array with chunking."""
        chunk_size = calculate_chunk_size(shape, dtype)
        return self.store.empty(name=name, shape=shape, chunks=(chunk_size), dtype=dtype)

    @staticmethod
    def calculate_chunk_size(shape, dtype):
        """Calculate optimal chunk size for Zarr arrays"""
        data_size_bytes = np.dtype(dtype).itemsize
        target_chunk_bytes = 1024 * 1024  # 1MB
        chunk_dim = int((target_chunk_bytes / data_size_bytes) ** (1/shape[1]))
        return tuple(chunk_dim for _ in range(shape[1]))


@dataclass
class ConfigData:
    """Holds configuration data."""
    entity_manager: EntityManager
    config: Any
    frequency_bands: FrequencyBands
    n_bands: int
    n_objs: int
    n_src: int = 0
    n_rays: int = 0
    n_points: int = 0

@dataclass
class AcousticRayTracer:
    """Main class for acoustic ray tracing."""
    entity_manager: EntityManager

    def __post_init__(self):
        self.config = self.entity_manager.get('config')
        self.frequency_bands = FrequencyBands(self.entity_manager)

        # Initialize data structures
        self.n_bands = len(self.frequency_bands.get_bands())
        self.n_objs = len(self.config.objects)
        self.n_rays = self.config.system.number_of_rays
        self.n_points = self.n_rays * self.n_bands

        # Scene and geometry
        self.scene = rtcs.EmbreeScene()
        self.geometry: Optional[GeometryData] = None
        self.material_properties: Optional[MaterialProperties] = None
        self.medium_properties: Optional[MediumProperties] = None

        # Ray and output data (using Zarr)
        self.ray_data = ZarrRayData(self.entity_manager)
        self.output_data = ZarrOutputData(self.entity_manager)

        # Recursion state
        self.recursion_idx = 0

        # Initialize
        self._initialize_scene()
        self._initialize_sources()
        self._initialize_outputs()
        self._initialize_objects()

        # Create Embree scene
        embree_mesh = TriangleMesh(self.scene, self.geometry.mesh_info)

    def _initialize_scene(self):
        """Initialize the acoustic domain scene."""
        ac_geometry = np.array(self.config.acoustic_domain.geometry)
        ac_max = np.max(ac_geometry, axis=0)
        ac_min = np.min(ac_geometry, axis=0)

        mesh = trimesh.creation.box(bounds=(ac_min, ac_max))
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32)
        
        mesh_info = mesh.vertices[mesh.faces]
        scene_info = np.full((mesh_info.shape[0],), [-1], dtype=np.int32)

        # Initialize medium properties
        sound_speed = self.config.acoustic_domain.acoustic_shader.sound_speed
        density = self.config.acoustic_domain.acoustic_shader.density
        temperature = self.config.acoustic_domain.acoustic_shader.temperature
        impedance = self.config.acoustic_domain.acoustic_shader.impedence

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
            absorption_coeffs=np.full((n_faces, self.n_bands), 1.0, dtype=np.float32),
            absorption_phases=np.full((n_faces, self.n_bands), 0.0, dtype=np.float32),
            reflection_coeffs=np.full((n_faces, self.n_bands), 0.0, dtype=np.float32),
            reflection_phases=np.full((n_faces, self.n_bands), 0.0, dtype=np.float32),
            refraction_coeffs=np.full((n_faces, self.n_bands), 0.0, dtype=np.float32),
            refraction_phases=np.full((n_faces, self.n_bands), 0.0, dtype=np.float32),
            scattering_coeffs=np.full((n_faces, self.n_bands), 0.0, dtype=np.float32),
            scattering_phases=np.full((n_faces, self.n_bands), 0.0, dtype=np.float32),
            roughness=np.full((n_faces, 1), 0.0, dtype=np.float32)
        )

        self.geometry = GeometryData(
            vertices=vertices,
            faces=faces,
            mesh_info=mesh_info,
            scene_info=scene_info
        )

    def _initialize_sources(self):
        """Initialize source positions and directions."""
        n_src = 0

        for src_config in self.config.sources:
            pose = np.load(f"{src_config.pose_path}/{src_config.name}.npz")
            source_pos = pose[pose.files[0]].reshape(-1, 3)
            n_src += 1

            source_bands = np.zeros((self.n_rays * self.n_bands, 1), dtype=np.int32)
            for idx in range(self.n_rays):
                lo_idx = self.n_bands * idx
                hi_idx = self.n_bands * (idx + 1)
                source_bands[lo_idx:hi_idx] = np.arange(self.n_bands).reshape(-1, 1)

            self.ray_data.origins_bands.append(source_bands, axis=0)

            source_idx = np.full((self.n_points, 1), [src_config.idx], dtype=np.int32)
            self.ray_data.origins_idx.append(source_idx, axis=0)

            source_arr = np.full((self.n_points, 3), [source_pos], dtype=np.float32)
            self.ray_data.origins.append(source_arr, axis=0)

        self.n_src = n_src

    def _initialize_outputs(self):
        """Initialize output positions."""
        out_idx = -2

        for out_config in self.config.outputs:
            out_idx += -1
            pose = np.load(f"{out_config.pose_path}/{out_config.name}.npz")
            output_pos = pose[pose.files[0]].reshape(-1, 3)

            output_arr = np.full((self.n_points, 3), [output_pos], dtype=np.float32)
            output_idx = np.full((self.n_points, 1), [out_config.idx], dtype=np.float32)

            self.ray_data.destinations.append(output_arr, axis=0)
            self.ray_data.destinations_idx.append(output_idx, axis=0)

            # Create output sphere geometry
            if out_config.size == 0:
                out_config.size = 0.1

            mesh = trimesh.creation.icosphere(subdivisions=2, radius=out_config.size)
            mesh.apply_transform([
                [1, 0, 0, output_pos[0][0]],
                [0, 1, 0, output_pos[00][1]],
                [0, 0, 1, output_pos[0][2]],
                [0, 0, 0, 1]
            ])

            vertices = mesh.vertices.astype(np.float32)
            faces = mesh.faces.astype(np.int32)

            self.geometry.mesh_info = np.append(self.geometry.mesh_info, mesh.vertices[mesh.faces], axis=0)
            self.geometry.scene_info = np.append(self.geometry.scene_info, np.full((mesh.vertices[mesh.faces].shape[0],), [out_idx], dtype=np.int32))

            # Add null properties for output geometry
            n_faces = vertices[faces].shape[0]
            self.material_properties.roughness = np.append(self.material_properties.roughness, np.full((n_faces, 1), 0.0, dtype=np.float32), axis=0)

            for prop_name in ['absorption', 'reflection', 'refraction', 'scattering']:
                coeffs = getattr(self.material_properties, f'{prop_name}_coeffs')
                phases = getattr(self.material_properties, f'{prop_name}_phases')

                new_coeffs = np.full((n_faces, self.n_bands),  1.0 if prop_name == 'absorption' else 0.0, dtype=np.float32)
                new_phases = np.full((n_faces, self.n_bands), 0.0, dtype=np.float32)

                setattr(self.material_properties, f'{prop_name}_coeffs', np.append(coeffs, new_coeffs, axis=0))
                setattr(self.material_properties, f'{prop_name}_phases', np.append(phases, new_phases, axis=0))

    def _initialize_objects(self):
        """Initialize scene objects."""
        for obj_config in self.config.objects:
            mesh_file = f"{obj_config.obj_path}/{obj_config.name}.npz"
            if not os.path.isfile(mesh_file):
                continue

            data = np.load(mesh_file, allow_pickle=False)
            vertices = data[data.files[0]].astype(np.float32)
            faces = data[data.files[2]].astype(np.int32)

            self.geometry.mesh_info = np.append(self.geometry.mesh_info, vertices[faces], axis=0)
            self.geometry.scene_info = np.append(self.geometry.scene_info, np.full((vertices[faces].shape[0],), [obj_config.idx], dtype=np.int32))

            # Get object acoustic properties
            n_faces = vertices[faces].shape[0]
            roughness = obj_config.acoustic_shader.roughness
            self.material_properties.roughness = np.append(self.material_properties.roughness, np.full((n_faces, 1), [roughness], dtype=np.float32), axis=0)

            for prop_name in ['absorption', 'reflection', 'refraction', 'scattering']:
                prop = getattr(obj_config.acoustic_shader.acoustic_properties, prop_name)
                coeffs, phases = prop.get_bands_avg(self.frequency_bands.get_bands())

                existing_coeffs = getattr(self.material_properties, f'{prop_name}_coeffs')
                existing_phases = getattr(self.material_properties, f'{prop_name}_phases')

                setattr(self.material_properties, f'{prop_name}_coeffs', np.append(existing_coeffs, np.full((n_faces, self.n_bands), coeffs, dtype=np.float32), axis=0))
                setattr(self.material_properties, f'{prop_name}_phases', np.append(existing_phases, np.full((n_faces, self.n_bands), phases, dtype=np.float32), axis=0))

    def _compute_acoustic_domain_coefficients(self, c: float, rho: float, T: float, Z: float):
        """Compute absorption and phase shift coefficients for air."""
        freqs = np.unique(self.frequency_bands.get_bands())[:-1]
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

    def _compute_object_coefficients(self, c: float, rho: float, E: float, nu: float, damping: float):
        """Calculate medium attenuation coefficient and phase shift for objects."""
        n_bands = self.n_bands
        alpha = np.zeros((n_bands, 1), dtype=np.float32)
        beta = np.zeros((n_bands, 1), dtype=np.float32)

        for idx in range(n_bands):
            min_freq, max_freq = self.frequency_bands.get_bands()[idx]
            alpha[idx], beta[idx] = _compute_rayleigh_damping(min_freq, max_freq, damping)

        freqs = np.unique(self.frequency_bands.get_bands())[:-1]
        omega = 2 * np.pi * freqs
        omega = omega.reshape(freqs.shape[0], 1)

        K = E / (3 * (1 - 2 * nu))
        G = E / (2 * (1 + nu))
        Z = rho * c

        is_solid = G > 0.1 * E

        alpha_attenuation = (alpha / (2 * c)) + (beta * omega**2 / (2 * c))

        if not is_solid:
            viscosity = 1.8e-5 if rho < 100 else 1e-3
            alpha_viscous = (2 * omega**2 * viscosity) / (3 * rho * c**3)
        else:
            alpha_viscous = 0

        alpha_attenuation = alpha_attenuation + alpha_viscous
        phase_shift = omega / c

        return alpha_attenuation, phase_shift

    def run(self):
        """Run the ray tracing simulation."""
        # Initialize directions
        self._initialize_directions()

        # Start recursive ray tracing
        self._ray_tracing_loop()

    def _initialize_directions(self):
        """Initialize ray directions using Fibonacci sphere distribution."""
        num_points = self.n_rays * self.n_src
        num_dirs = self.n_rays * self.n_src * self.n_bands

        directions = np.zeros((num_dirs, 3), dtype=np.float32)

        # Fibonacci sphere
        phi = np.pi * (3. - np.sqrt(5.))
        theta = phi * np.arange(num_points)
        z = np.linspace(1/num_points - 1, 1 - 1/num_points, num_points)
        radius = np.sqrt(1 - z * z)
        y = radius * np.sin(theta)
        x = radius * np.cos(theta)

        dirs = np.array(list(zip(x, y, z)), dtype=np.float32)

        for idx in range(self.n_rays):
            lo_idx = self.n_bands * idx
            hi_idx = self.n_bands * (idx + 1)
            directions[lo_idx:hi_idx] = dirs[idx]

        # Set initial directions towards destinations
        main_dirs = self.ray_data.destinations[:] - self.ray_data.origins[:]
        main_dirs_norm = np.linalg.norm(main_dirs, axis=1, keepdims=True)
        main_dirs_norm[main_dirs_norm <= 1e-10] = 1e-10
        main_dirs = main_dirs / main_dirs_norm

        for idx in range(self.n_src):
            main_idx = self.n_rays * self.n_bands * idx
            directions[main_idx] = main_dirs[main_idx]

        self.ray_data.directions.append(directions, axis=0)
        self.ray_data.energies.append(np.full((num_dirs, 1), 1.0, dtype=np.float32), axis=0)
        self.ray_data.phases.append(np.full((num_dirs, 1), 0.0, dtype=np.float32), axis=0)
        self.ray_data.delay.append(np.full((num_dirs, 1), 0.0, dtype=np.float32), axis=0)

    def _ray_tracing_loop(self):
        """Recursive ray tracing loop."""
        t1 = time.time()

        # Trace rays using Embree
        res = self.scene.run(self.ray_data.origins[:].astype(np.float32), self.ray_data.directions[:].astype(np.float32), output=1)

        t2 = time.time()
        print(f"Ran in {t2 - t1:.3f} s")

        ray_inter = res["geomID"] >= 0 
        print(f"{sum(ray_inter)} rays intersect geometry (over {self.ray_data.origins.shape[0]})")

        if not np.any(ray_inter):
            self._finalize()
            return

        # Process intersections
        self._process_intersections(res, ray_inter)

    def _process_intersections(self, res: Dict, ray_inter: np.ndarray):
        """Process ray intersection results."""
        primID = res["primID"][ray_inter]
        u = res["u"][ray_inter]
        v = res["v"][ray_inter]
        w = 1 - u - v

        a = self.geometry.mesh_info[primID][:, 0, :]
        b = self.geometry.mesh_info[primID][:, 1, :]
        c = self.geometry.mesh_info[primID][:, 2, :]

        inters = (np.vstack(w) * a + np.vstack(u) * b + np.vstack(v) * c)

        # Save ray data
        self._save_ray_data(self.ray_data.origins, inters)

        # Filter rays that intersect
        self._filter_intersected_rays(ray_inter, primID, inters)

    def _save_ray_data(self, origins: np.ndarray, hit_points: np.ndarray):
        """Save ray data to JSON file."""
        data_dict = {
            'origins': origins[:].tolist(),
            'hit_points': hit_points[:].tolist()
        }

        os.makedirs('ray_datas', exist_ok=True)
        filepath = f"ray_datas/embreex_{self.recursion_idx:04}.json"

        with open(filepath, 'w') as f:
            json.dump(data_dict, f, indent=2)

    def _filter_intersected_rays(self, ray_inter: np.ndarray, primID: np.ndarray,
                                  inters: np.ndarray):
        """Filter and process intersected rays."""
        self.ray_data.apply_mask(ray_inter)

        # Compute path length
        path_length = np.sqrt(np.sum((inters - self.ray_data.origins[:])**2, axis=1)).reshape(-1, 1)

        # Update medium properties
        self._update_medium_properties(path_length)

        # Get object indices and filter
        hits_obj_idx = self.geometry.scene_info[primID]
        intersect_mask = hits_obj_idx >= 0

        # Collect output data
        self._collect_output_data(hits_obj_idx, intersect_mask, path_length)

        # Continue with remaining rays
        if np.any(intersect_mask):
            self._continue_tracing(inters, primID, path_length, intersect_mask)

    def _update_medium_properties(self, path_length: np.ndarray):
        """Update medium attenuation and phase shift."""
        n_rays = self.ray_data.origins.shape[0]

        # Default medium properties (air)
        medium_speed = np.full((n_rays, 1), self.medium_properties.speed, dtype=np.float32)
        medium_alpha = np.full((n_rays, self.n_bands), self.medium_properties.alpha, dtype=np.float32)
        medium_beta = np.full((n_rays, self.n_bands), self.medium_properties.beta, dtype=np.float32)

        # Check for objects containing origins
        for obj_config in self.config.objects:
            mesh_file = f"{obj_config.obj_path}/{obj_config.name}.npz"
            if not os.path.isfile(mesh_file):
                continue

            data = np.load(mesh_file, allow_pickle=False)
            vertices = data[data.files[0]].astype(np.float32)
            vertex_normals = data[data.files[1]].astype(np.float32)
            faces = data[data.files[2]].astype(np.int32)

            mesh = trimesh.Trimesh(vertices=vertices, vertex_normals=vertex_normals, faces=faces)

            medium_mask = mesh.contains(self.ray_data.origins[:])
            if np.any(medium_mask):
                print(f'Find medium object properties for {obj_config.name}')

                sound_speed = obj_config.acoustic_shader.sound_speed
                density = obj_config.acoustic_shader.density
                young_modulus = obj_config.acoustic_shader.young_modulus
                poisson_ratio = obj_config.acoustic_shader.poisson_ratio
                damping = obj_config.acoustic_shader.damping

                alpha, beta = self._compute_object_coefficients(sound_speed, density, young_modulus, poisson_ratio, damping)

                medium_speed[medium_mask] = sound_speed
                medium_alpha[medium_mask] = alpha.T
                medium_beta[medium_mask] = beta.T

        # Apply medium attenuation
        origins_bands_idx = np.arange(self.ray_data.origins_bands[:].T.shape[0])
        attenuation = np.exp(-medium_alpha * path_length)
        new_energies = self.ray_data.energies[:] * attenuation[origins_bands_idx, self.ray_data.origins_bands[:]]
        self.ray_data.energies.set_mask_selection(np.ones_like(self.ray_data.energies, dtype=bool), new_energies.flatten())
        del new_energies

        # Apply phase shift
        phase_shift = path_length * medium_beta[origins_bands_idx, self.ray_data.origins_bands[:]]
        new_phases = (self.ray_data.phases[:] + phase_shift) % (2 * np.pi)
        self.ray_data.phases.set_mask_selection(np.ones_like(self.ray_data.phases, dtype=bool), new_phases.flatten())

        # Update delay
        new_delay = path_length / medium_speed
        new_delay = self.ray_data.delay[:] + new_delay
        self.ray_data.delay.set_mask_selection(np.ones_like(self.ray_data.delay, dtype=bool), new_delay.flatten())

    def _collect_output_data(self, hits_obj_idx: np.ndarray, intersect_mask: np.ndarray, path_length: np.ndarray):
        """Collect rays that reached output destinations."""
        output_mask = hits_obj_idx <= -3

        if np.any(output_mask):
            self.output_data.source.append(self.ray_data.origins_idx[:][output_mask].astype(np.int32), axis=0)
            self.output_data.bands.append(self.ray_data.origins_bands[:][output_mask].astype(np.int32), axis=0)
            self.output_data.energies.append(self.ray_data.energies[:][output_mask].astype(np.float32), axis=0)
            self.output_data.phases.append(self.ray_data.phases[:][output_mask].astype(np.float32), axis=0)
            self.output_data.delay.append(self.ray_data.delay[:][output_mask].astype(np.float32), axis=0)
            self.output_data.origins.append(self.ray_data.origins[:][output_mask].astype(np.float32), axis=0)
            self.output_data.destinations.append(self.ray_data.destinations[:][output_mask].astype(np.float32), axis=0)
            self.output_data.destinations_idx.append(self.ray_data.destinations_idx[:][output_mask].astype(np.int32), axis=0)
            self.output_data.directions.append(self.ray_data.directions[:][output_mask].astype(np.float32), axis=0)
            print(f'Output: {np.count_nonzero(output_mask)}, 'f'{self.output_data.energies[:].shape[0]}')

    def _continue_tracing(self, inters: np.ndarray, primID: np.ndarray, path_length: np.ndarray, intersect_mask: np.ndarray):
        """Continue ray tracing for rays that hit objects."""
        # Filter for rays that hit objects
        inters = inters[intersect_mask]
        path_length = path_length[intersect_mask]

        # Compute normals
        a = self.geometry.mesh_info[primID][:, 0, :]
        b = self.geometry.mesh_info[primID][:, 1, :]
        c = self.geometry.mesh_info[primID][:, 2, :]

        normals = np.cross(b - a, c - a)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals[intersect_mask]

        # Filter remaining data with intersect_mask
        self.ray_data.apply_mask(intersect_mask)

        # Get material properties for intersected faces
        self._apply_material_properties(primID, inters, intersect_mask, normals)

    def _apply_material_properties(self, primID: np.ndarray, inters: np.ndarray, intersect_mask: np.ndarray, normals: np.ndarray):
        """Apply material properties at intersection points."""
        primID_filtered = primID[intersect_mask]

        # Get material properties
        origins_bands_idx = np.arange(self.ray_data.origins_bands[:].T.shape[0])
        bands = self.ray_data.origins_bands[:].flatten()

        abs_coeffs = self.material_properties.absorption_coeffs[primID_filtered][origins_bands_idx, bands]
        abs_phases = self.material_properties.absorption_phases[primID_filtered][origins_bands_idx, bands]
        refl_coeffs = self.material_properties.reflection_coeffs[primID_filtered][origins_bands_idx, bands]
        refl_phases = self.material_properties.reflection_phases[primID_filtered][origins_bands_idx, bands]
        scat_coeffs = self.material_properties.scattering_coeffs[primID_filtered][origins_bands_idx, bands]
        scat_phases = self.material_properties.scattering_phases[primID_filtered][origins_bands_idx, bands]

        # Compute incident angles
        dot_projection = np.sum(self.ray_data.directions[:] * normals, axis=1)
        incident_angles = np.arccos(-dot_projection)
        angle_factor = np.cos(incident_angles)
        angle_factor[angle_factor == 0] = 1e-16

        # Compute reflection and scattering
        reflected_energies = self.ray_data.energies[:] * refl_coeffs.reshape(-1, 1)
        reflected_phases = self.ray_data.phases[:] * -refl_phases.reshape(-1, 1) % (2 * np.pi)

        # Compute new origins
        new_origins = inters + (0.01 * normals)
        new_origins = new_origins.astype(np.float32)

        # Compute reflection directions
        reflected_directions = self._compute_reflection_directions(self.ray_data.directions[:], normals, incident_angles)

        # Generate scattering rays
        scattered_data = self._generate_scattering_rays(new_origins, normals, scat_coeffs)

        # Combine reflected and scattered rays
        temp_origins = np.append(new_origins, scattered_data['origins'], axis=0)
        temp_directions = np.append(reflected_directions, scattered_data['directions'], axis=0)
        temp_energies = np.append(reflected_energies, scattered_data['energies'], axis=0)
        temp_phases = np.append(reflected_phases, scattered_data['phases'], axis=0)

        # resize zarr array
        new_size = temp_origins.shape[0]
        self.ray_data.origins.resize((new_size, self.ray_data.origins.shape[1]))
        self.ray_data.directions.resize((new_size, self.ray_data.directions.shape[1]))
        self.ray_data.energies.resize((new_size, self.ray_data.energies.shape[1]))
        self.ray_data.phases.resize((new_size, self.ray_data.phases.shape[1]))

        # Rewrite zarr array with combined reflected and scattered rays
        self.ray_data.origins.set_mask_selection(np.ones_like(self.ray_data.origins, dtype=bool), temp_origins.flatten())
        self.ray_data.directions.set_mask_selection(np.ones_like(self.ray_data.directions, dtype=bool), temp_directions.flatten())
        self.ray_data.energies.set_mask_selection(np.ones_like(self.ray_data.energies, dtype=bool), temp_energies.flatten())
        self.ray_data.phases.set_mask_selection(np.ones_like(self.ray_data.phases, dtype=bool), temp_phases.flatten())

        # Append remaining scattered_data
        self.ray_data.origins_idx.append(scattered_data['origins_idx'], axis=0)
        self.ray_data.origins_bands.append(scattered_data['origins_bands'], axis=0)
        self.ray_data.destinations.append(scattered_data['destinations'], axis=0)
        self.ray_data.destinations_idx.append(scattered_data['destinations_idx'], axis=0)
        self.ray_data.delay.append(scattered_data['delay'], axis=0)

        # Apply energy threshold
        self._apply_energy_threshold()

        # Recursive call
        self.recursion_idx += 1
        if self.ray_data.origins.shape[0] > 0:
            self._ray_tracing_loop()
        else:
            self._finalize()

    def _compute_reflection_directions(self, incident_directions: np.ndarray, normals: np.ndarray, incident_angles: np.ndarray) -> np.ndarray:
        """Compute reflection direction vectors."""
        # Normalize inputs
        incident_directions = incident_directions / np.linalg.norm(incident_directions, axis=1, keepdims=True)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        # Compute components
        n_dot_i = np.sum(normals * incident_directions, axis=1, keepdims=True)
        incident_normals = n_dot_i * normals
        incident_tangent = incident_directions - incident_normals

        # Normalize tangent
        tangent_norm = np.linalg.norm(incident_tangent, axis=1, keepdims=True)
        incident_tangent_unit = incident_tangent / (tangent_norm + 1e-10)

        # Compute reflection direction
        reflection_directions = (np.cos(incident_angles.reshape(-1, 1)) * incident_normals - np.sin(incident_angles.reshape(-1, 1)) * incident_tangent_unit)

        return reflection_directions / np.linalg.norm(reflection_directions, axis=1, keepdims=True)

    def _generate_scattering_rays(self, origins: np.ndarray, normals: np.ndarray, scat_coeffs: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate scattering rays on hemisphere."""
        n_scat_origins = origins.shape[0]
        max_scattering = self.config.interface.max_scattering

        # Generate number of scattering rays
        roughness = self.material_properties.roughness
        n_scat_rays = np.random.randint(1, max_scattering, size=(n_scat_origins, 1))
        n_samples = np.sum(n_scat_rays)

        # Initialize arrays
        result = {
            'origins': np.zeros((n_samples, 3), dtype=np.float32),
            'origins_idx': np.zeros((n_samples, 1), dtype=np.int32),
            'origins_bands': np.zeros((n_samples, 1), dtype=np.int32),
            'destinations': np.zeros((n_samples, 3), dtype=np.float32),
            'destinations_idx': np.zeros((n_samples, 1), dtype=np.int32),
            'directions': np.zeros((n_samples, 3), dtype=np.float32),
            'normals': np.zeros((n_samples, 3), dtype=np.float32),
            'energies': np.zeros((n_samples, 1), dtype=np.float32),
            'phases': np.zeros((n_samples, 1), dtype=np.float32),
            'delay': np.zeros((n_samples, 1), dtype=np.float32)
        }

        # Generate random directions on hemisphere
        hi_idx = 0
        for idx in range(n_scat_origins):
            lo_idx = hi_idx
            hi_idx = lo_idx + int(n_scat_rays[idx])
            n_rays_this = hi_idx - lo_idx

            # Copy info array
            result['origins'][lo_idx:hi_idx] = origins[idx]
            result['origins_idx'][lo_idx:hi_idx] = self.ray_data.origins_idx[idx]
            result['origins_bands'][lo_idx:hi_idx] = self.ray_data.origins_bands[idx]
            result['destinations'][lo_idx:hi_idx] = self.ray_data.destinations[idx]
            result['destinations_idx'][lo_idx:hi_idx] = self.ray_data.destinations_idx[idx]
            result['directions'][lo_idx:hi_idx] = self.ray_data.directions[idx]
            result['normals'][lo_idx:hi_idx] = normals[idx]
            result['energies'][lo_idx:hi_idx] = self.ray_data.energies[idx]
            result['phases'][lo_idx:hi_idx] = self.ray_data.phases[idx]
            result['delay'][lo_idx:hi_idx] = self.ray_data.delay[idx]

            # Generate random directions on hemisphere
            random_dirs = np.random.uniform(-1, 1, (n_rays_this, 3))
            random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)

            # Ensure directions point along hemisphere oriented by normal
            normal = normals[idx]
            dot_products = np.sum(random_dirs * normal, axis=1)
            flip_mask = dot_products < 0
            random_dirs[flip_mask] = -random_dirs[flip_mask]

            result['directions'][lo_idx:hi_idx] = random_dirs

            # Distribute energy among scattering rays
            result['energies'][lo_idx:hi_idx] = scat_coeffs[idx] / n_rays_this

        return result

    def _apply_energy_threshold(self):
        """Apply energy threshold to terminate low-energy rays."""
        termination_energy = 1e-16
        termination_mask = self.ray_data.energies[:] > termination_energy
        termination_mask = termination_mask.reshape(-1,)

        self.ray_data.apply_mask(termination_mask)

        n_terminated = np.count_nonzero(~termination_mask)
        if n_terminated > 0:
            print(f'Terminated {n_terminated} rays below energy threshold')

    def _finalize(self):
        """Finalize ray tracing and generate impulse responses."""
        print("Ray tracing complete. Generating impulse responses...")

        # Compute and save ambisonic impulse responses
        self._compute_and_save_ir()

        # Convolve with audio files if configured
        self._convolve_with_audio()

    def _compute_and_save_ir(self):
        """Compute and save ambisonic impulse responses."""
        sample_rate = int(self.config.system.sample_rate)
        n_src = len(self.config.sources)
        n_outs = len(self.config.outputs)
        n_bands = self.n_bands
        freq_bands = self.frequency_bands.get_bands()

        # Sort output by delay
        sort_idx = np.argsort(self.output_data.delay[:].flatten())

        delay_sorted = self.output_data.delay[:].flatten()[sort_idx]
        energies_sorted = self.output_data.energies[:].flatten()[sort_idx]
        phases_sorted = self.output_data.phases[:].flatten()[sort_idx]
        sources_sorted = self.output_data.source[:].flatten()[sort_idx]
        bands_sorted = self.output_data.bands[:].flatten()[sort_idx]
        directions_sorted = self.output_data.directions[:][sort_idx]

        # Convert delay to samples
        delay_samples = np.round(delay_sorted * sample_rate).astype(int)

        # Determine max IR length
        ir_length = int(np.ceil(np.max(delay_samples))) + 10

        # Process each source-output-band combination
        for src_idx in range(n_src):
            for out_idx in range(n_outs):
                for bands_idx in range(n_bands):
                    # Get ambisonic order for this output
                    ambisonic_order = 1  # default
                    for out_config in self.config.outputs:
                        if out_config.idx == out_idx:
                            ambisonic_order = out_config.order
                            break

                    n_channels = (ambisonic_order + 1) ** 2
                    ambisonics_ir = np.zeros((n_channels, ir_length), dtype=np.float32)

                    # Filter by source, output, and band
                    mask = (
                        (sources_sorted == src_idx) &
                        (bands_sorted == bands_idx)
                    )

                    if not np.any(mask):
                        continue

                    # Get filtered data
                    filtered_energies = energies_sorted[mask]
                    filtered_phases = phases_sorted[mask]
                    filtered_delay = delay_samples[mask]
                    filtered_directions = directions_sorted[mask]

                    # Compute complex amplitudes
                    complex_amplitudes = np.sqrt(filtered_energies) * np.exp(1j * filtered_phases)

                    # Convert directions to spherical coordinates
                    x, y, z = filtered_directions[:, 0], filtered_directions[:, 1], filtered_directions[:, 2]
                    theta = np.arctan2(y, x)  # Azimuth
                    phi = np.arcsin(z)  # Elevation

                    # Compute spherical harmonics for each order
                    self._compute_spherical_harmonics(ambisonics_ir, filtered_delay, complex_amplitudes, theta, phi, ambisonic_order)

                    # Apply windowing
                    window = np.hanning(ir_length)
                    for ch in range(n_channels):
                        ambisonics_ir[ch] *= window

                    # Normalize
                    max_val = np.max(np.abs(ambisonics_ir))
                    if max_val > 0:
                        ambisonics_ir /= max_val

                    # Save impulse response
                    output_dir = 'impulse_response'
                    os.makedirs(output_dir, exist_ok=True)

                    filename = f"{output_dir}/aIR_{src_idx}_{out_idx}_{bands_idx}.wav"
                    sf.write(filename, ambisonics_ir.T, sample_rate, subtype='FLOAT')

                    print(f"Saved IR: {filename}")
                    print(f"  Shape: {ambisonics_ir.T.shape} (samples, channels)")
                    print(f"  Duration: {ir_length / sample_rate:.2f} seconds")

    def _compute_spherical_harmonics(self, ambisonics_ir: np.ndarray, delay_samples: np.ndarray, complex_amplitudes: np.ndarray, theta: np.ndarray, phi: np.ndarray, ambisonic_order: int):
        """Compute spherical harmonics and add to IR buffer."""
        n_rays = len(delay_samples)

        if ambisonic_order >= 0:
            # Order 0: W channel (omnidirectional)
            Y_00 = 1.0 / np.sqrt(4 * np.pi)
            for i in range(n_rays):
                sample_idx = delay_samples[i]
                ambisonics_ir[0, sample_idx] += np.real(complex_amplitudes[i] * Y_00)

        if ambisonic_order >= 1:
            # Order 1: X, Y, Z channels (ACN ordering)
            Y_1n1 = np.sqrt(3/(4*np.pi)) * np.sin(theta) * np.cos(phi)  # Y
            Y_10 = np.sqrt(3/(4*np.pi)) * np.sin(phi)                   # Z
            Y_11 = np.sqrt(3/(4*np.pi)) * np.cos(theta) * np.cos(phi)   # X

            for i in range(n_rays):
                sample_idx = delay_samples[i]
                ambisonics_ir[1, sample_idx] += np.real(complex_amplitudes[i] * Y_1n1[i])
                ambisonics_ir[2, sample_idx] += np.real(complex_amplitudes[i] * Y_10[i])
                ambisonics_ir[3, sample_idx] += np.real(complex_amplitudes[i] * Y_11[i])

        if ambisonic_order >= 2:
            # Order 2: Additional 5 channels
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            sqrt_15_4pi = np.sqrt(15/(4*np.pi))

            Y_2n2 = sqrt_15_4pi * sin_theta * cos_theta * cos_phi**2      # R
            Y_2n1 = sqrt_15_4pi * sin_theta * sin_phi * cos_phi           # S
            Y_20 = np.sqrt(5/(16*np.pi)) * (3*sin_phi**2 - 1)              # T
            Y_21 = sqrt_15_4pi * cos_theta * sin_phi * cos_phi             # U
            Y_22 = sqrt_15_4pi * (cos_theta**2 - sin_theta**2) * cos_phi**2  # V

            for i in range(n_rays):
                sample_idx = delay_samples[i]
                ambisonics_ir[4, sample_idx] += np.real(complex_amplitudes[i] * Y_2n2[i])
                ambisonics_ir[5, sample_idx] += np.real(complex_amplitudes[i] * Y_2n1[i])
                ambisonics_ir[6, sample_idx] += np.real(complex_amplitudes[i] * Y_20[i])
                ambisonics_ir[7, sample_idx] += np.real(complex_amplitudes[i] * Y_21[i])
                ambisonics_ir[8, sample_idx] += np.real(complex_amplitudes[i] * Y_22[i])

    def _convolve_with_audio(self):
        """Convolve impulse responses with audio files."""
        sample_rate = int(self.config.system.sample_rate)
        output_dir = 'ambisonics_output'

        for out_config in self.config.outputs:
            for src_config in self.config.sources:
                if not (hasattr(src_config, 'audio_file') and
                        os.path.exists(src_config.audio_file)):
                    continue

                file_audio = src_config.audio_file

                # Convert audio to multibands
                multi_bands_audio = _mono_to_bands(file_audio, sample_rate, self.frequency_bands.get_bands())

                # Convolve each band
                for bands_idx in range(multi_bands_audio.shape[0]):
                    # Load band-specific IR
                    ir_filename = f"impulse_response/aIR_{src_config.idx}_{out_config.idx}_{bands_idx}.wav"
                    ambisonics_ir, sr = sf.read(ir_filename)

                    # Convolve
                    mono_audio = multi_bands_audio[bands_idx]
                    ambisonics_output = self._convolve_mono_to_ambisonics(mono_audio, ambisonics_ir.T)

                    # Normalize
                    ambisonics_output_normalized = self._normalize_ambisonics_output(ambisonics_output)

                    # Save
                    os.makedirs(output_dir, exist_ok=True)
                    filename = f"{output_dir}/{src_config.idx}_{out_config.idx}_{bands_idx}.wav"
                    sf.write(filename, ambisonics_output_normalized.T, sample_rate, subtype='FLOAT')

                    print(f"Saved ambisonics output: {filename}")
                    print(f"  Shape: {ambisonics_output_normalized.T.shape}")

    def _convolve_mono_to_ambisonics(self, mono_audio: np.ndarray, ambisonics_ir: np.ndarray, method: str = 'fft') -> np.ndarray:
        """Convolve mono audio with ambisonics IR."""
        n_channels, n_ir_samples = ambisonics_ir.shape
        n_audio_samples = len(mono_audio)
        output_length = n_audio_samples + n_ir_samples - 1

        ambisonics_output = np.zeros((n_channels, output_length), dtype=np.float32)

        for ch in range(n_channels):
            if method == 'fft':
                ambisonics_output[ch] = fftconvolve(mono_audio, ambisonics_ir[ch], mode='full')
            else:
                ambisonics_output[ch] = convolve(mono_audio, ambisonics_ir[ch], mode='full')

        return ambisonics_output

    def _normalize_ambisonics_output(self, ambisonics_output: np.ndarray, normalize_individually: bool = False) -> np.ndarray:
        """Normalize ambisonics output to prevent clipping."""
        if normalize_individually:
            max_vals = np.max(np.abs(ambisonics_output), axis=1, keepdims=True)
            max_vals[max_vals == 0] = 1.0
            normalized = ambisonics_output / max_vals
        else:
            max_val = np.max(np.abs(ambisonics_output))
            if max_val > 0:
                normalized = ambisonics_output / max_val
            else:
                normalized = ambisonics_output

        return normalized.astype(np.float32)

def main():
    """Main entry point for the acoustic ray tracer."""
    import argparse

    parser = argparse.ArgumentParser(description='Acoustic Ray Tracing Simulation')
    parser.add_argument('config', type=str, help='Path to configuration JSON file')
    args = parser.parse_args()

    # Create and run ray tracer tracer
    entity_manager = EntityManager(args.config)
    tracer = AcousticRayTracer(entity_manager)
    tracer.run()


if __name__ == "__main__":
    main()

