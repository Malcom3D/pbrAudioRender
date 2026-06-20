# code not used for now

import sys, os
sys.path.append(os.getcwd())
sys.path.append('../src/pbrAudioRender')
#sys.path.append('../src')

#from pbrAudioRender import pbrAudioRender
#config_file = 'config.json'
#render = pbrAudioRender(config_file)
###########################################################
import sys, os
sys.path.append(os.getcwd())
sys.path.append('../src/pbrAudioRender')

import dask.array as da
from dask import delayed, compute

from pbrAudioCommon import np

from lib.soxel import Soxel
from lib.acoustic_field import AcousticField, FrequencyLimitedField, VelocityVectors
from lib.acoustic_shader import AcousticCoefficients, AcousticProperties, AcousticShader

freq_abs = np.array([20, 100, 500, 1000, 5000, 20000])
coeffs_abs = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
absorption = AcousticCoefficients(frequencies=freq_abs, coefficients=coeffs_abs)
freq_refl = np.array([20, 100, 500, 1000, 5000, 20000])
coeffs_refl = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.65])
reflection = AcousticCoefficients(frequencies=freq_refl, coefficients=coeffs_refl)
freq_sca = np.array([125, 250, 500, 1000, 2000, 4000])
coeffs_sca = np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
scattering = AcousticCoefficients(frequencies=freq_sca, coefficients=coeffs_sca)
prop = AcousticProperties(absorption=absorption, reflection=reflection, scattering=scattering)

sound_speed = 343.4
density = 1225
acusha = AcousticShader(sound_speed, density, prop)
vel = VelocityVectors(vx=1.324, vy=3.345, vz=1.2342)
acufield = AcousticField()
acufield.add_field(20,25,10,vel)
soxel = Soxel(0,2,acufield.get_field(20,25), acusha)
soxels = np.empty([100,100,100], dtype=object)
soxel_grid = da.from_array(soxels, chunks=(10,100,100))

@delayed
def write_soxel(i,j,k, soxel):
    return soxel

@delayed
def read_soxel(soxel_grid, i,j,k):
    print(soxel_grid[i,j,k].compute().item().input_pressures)

result = []
for i in range(soxel_grid.shape[0]):
    for j in range(soxel_grid.shape[1]):
        for k in range(soxel_grid.shape[2]):
            result.append(write_soxel(i,j,k, soxel))
# Compute in parallel
computed_results = compute(*results)
# Reshape back to original grid shape
soxel_grid = np.array(computed_results).reshape(soxel_grid.shape)

result = []
for i in range(soxel_grid.shape[0]):
    for j in range(soxel_grid.shape[1]):
        for k in range(soxel_grid.shape[2]):
            if soxel_grid.soxels[i,j,k].type == 2:
                print(i,j,k, soxel_grid[i,j,k].compute().item().input_pressures)

# Compute in parallel
compute(*results)

#####################################################################
import sys, os
sys.path.append(os.getcwd())
sys.path.append('../src/pbrAudioRender')
config_file = 'config.json'
from core.entity_manager import EntityManager
em = EntityManager(config_file)
from core.acoustic_engine import AcousticEngine
ac = AcousticEngine(em)
ac.update()

frequencies = em.get('frequency_bands')
bands = frequencies.get_bands()

config = em.get('config')
soxel_grid = em.get('soxel_grid') 
wave = em.get('wave_propagators',0)
lm = wave.layer_manager
frames = em.get('frames')
frames.next()
ac.update()
for i in range(soxel_grid.shape[0]):
    for j in range(soxel_grid.shape[1]):
        for k in range(soxel_grid.shape[2]):
            if soxel_grid.soxels[i,j,k].type == 1:
                for id in lm.layers.keys():
                    print(id, i,j,k, lm.layers[id].field[i,j,k])
                    if not lm.layers[id].field[i,j,k] == 0:
                        print(id, i,j,k, lm.layers[id].field[i,j,k])




import sys, os
sys.path.append(os.getcwd())
sys.path.append('../src/pbrAudioRender')

config_file = 'config.json'
from core.entity_manager import EntityManager
em = EntityManager(config_file)

from core.acoustic_engine import AcousticEngine
ac = AcousticEngine(em)

ac.update()

wave = em.get('wave_propagators',0)
lm = wave.layer_manager
lm.add_new('fdtd', 0)
lm.add_new('fdtd', 1)
lm.add_new('fdtd', 2)
lm.len_by_name('fdtd')
layer = lm.layers[0]

config = em.get('config')
soxel_grid = em.get('soxel_grid')
for i in range(soxel_grid.shape[0]):
    for j in range(soxel_grid.shape[1]):
        for k in range(soxel_grid.shape[2]):
            if soxel_grid.soxels[i,j,k].type == 2:
                print(i,j,k)

soxel_grid = em.get('soxel_grid')
config = em.get('config')
for source_config in config.sources:
    source = em.get('sources', source_config.idx)
    soxel_list = source.get_soxels()
    for i,j,k, soxel in soxel_list:
        print(i,j,k, soxel)
        soxel_grid.soxels[coord] = soxel

config = em.get('config')
soxel_grid = em.get('soxel_grid')
wave = em.get('wave_propagators',0)
lm = wave.layer_manager

for i in range(soxel_grid.shape[0]):
    for j in range(soxel_grid.shape[1]):
        for k in range(soxel_grid.shape[2]):
            if soxel_grid.soxels[i,j,k].type == 1:
                print(i,j,k, soxel_grid.soxels[i,j,k].input_pressures)


config = em.get('config')
for obj in config.objects:
    obj = em.get('objects', obj.idx)
    obj.get_soxels()
    soxel_list = obj.get_soxels()
    for coord, soxel in soxel_list:
        self.soxels[coord] = soxel

soxel_grid = em.get('soxel_grid')
shm_pressure = soxel_grid.get_shm_array('pressure', low_freq=1436.7514218360097, high_freq=1478.8514519965222)

type(shm_pressure)
shm_pressure.shape

for x in range(shm_pressure.shape[0]):
    for y in range(shm_pressure.shape[1]):
        for z in range(shm_pressure.shape[2]):
            if not shm_pressure[x,y,z] == 0:
               print(shm_pressure[x,y,z])

for x in range(shm_pressure.shape[0]):
    for y in range(shm_pressure.shape[1]):
        for z in range(shm_pressure.shape[2]):
            if not soxel_grid.soxels[x,y,z].input_pressures == None:
                shm_pressure[x,y,z] = soxel_grid.soxels[x,y,z].input_pressures.field[0].pressure
            else:
                shm_pressure[x,y,z] = 0.


from lib.frames import FrameCounter
frames = FrameCounter(em)
em.register('frames', frames)

from sources.spherical_source import SphericalSource
from sources.planar_source import PlanarSource

config = em.get('config')

source_map = {
    'spherical': SphericalSource,
    'planar': PlanarSource
}

entities_map = {
    'sources': ['SphericalSource', 'PlaneSource'],
    'objects': ['AcousticObject'],
    'outputs': ['OmnidirectionalOutput', 'Figure8Output', 'CardioidOutput', 'HypercardioidOutput'],
    'wave_propagators': 'WavePropagator',
    'layer_managers': 'LayerManager'
}

for source_config in config.sources:
    source = source_map.get(source_config.type)(em, source_config.idx)
    em.register('sources', source, source_config.idx)

for source_config in config.sources:
    obj = entities_map.get(source_config.type)(em, source_config.idx)
    for key in self.entities_map.keys():
        if 'sources' in key:
            for sub in self.entities_map[key]:
                if sub in type(obj):
                    entities = eval(f"self._{key}")
                    entities[idx] = obj






from lib.acoustic_field import VelocityVectors, FrequencyLimitedField, AcousticField
from pbrAudioCommon import np

vel = VelocityVectors(x=23.4, y=12.5, z=9.23)
ac = AcousticField()
ac.add_field(low_freq=10, high_freq=20, pressure=5, velocity=vel)
soxels = np.empty([10,10,10], dtype=object)
soxels[0,0,0] = ac

print(soxels[0,0,0])











from core.acoustic_engine import AcousticEngine
ac = AcousticEngine(em)

type(ac.frequencies)

from lib.frames import FrameCounter
frames = FrameCounter()
em.register_object('current_frame', frames)

from utils.gpu_acceleration import GPUManager
gpu_manager = GPUManager(em)
em.register_object('gpu', gpu_manager)

em.get_objects()


from lib.frames import FrameCounter
frames = FrameCounter(em)
em.register('frames', frames)

from sources.spherical_source import SphericalSource
from sources.planar_source import PlanarSource

config = em.get('config')

source_map = {
    'spherical': SphericalSource,
    'planar': PlanarSource
}

for source_config in config.sources:
   source = source_map.get(source_config.type)(em, source_config.idx)
   em.register('sources', source, source_config.idx)



import sys, os
sys.path.append(os.getcwd())
sys.path.append('../src/pbrAudioRender')
config_file = 'config.json'
from utils.config import Config


import sys, os
from pbrAudioCommon import np
sys.path.append(os.getcwd())
from typing import List, Dict, Any, Optional, Tuple, Union

from core.entity_manager import EntityManager

config_file = 'config.json'
entity_manager = EntityManager(config_file)

from utils.config import Config
from utils.gpu_acceleration import GPUManager
from lib.frames import FrameCounter
from core.soxel_grid import SoxelGrid
from core.acoustic_engine import AcousticEngine
from engine.wave_propagation import WavePropagation

gpu_manager = GPUManager(config)
frames = FrameCounter() if config.system.frame_limit == None else FrameCounter(frame_limit=config.system.frame_limit)
entity_manager = EntityManager(config, frames)
soxel_grid = SoxelGrid(config, entity_manager, frames)
acoustic_engine = AcousticEngine(config, gpu_manager, frames, soxel_grid, entity_manager)

acoustic_engine.wave_propagation.fdtd_solvers[0].update()
acoustic_engine.wave_propagation.fdtd_solvers[0].layer_manager.layers[0].grid[0,0,0]

#wave_propagation = WavePropagation(config, gpu_manager, frames, soxel_grid)

for source in config.sources:
    entity_manager.add_new(source)

entity_manager.get_soxels()

entity_manager.get_soxels()
entity_manager.objects[0].get_soxels()
entity_manager.objects[1].get_soxels()
entity_manager.sources[0].get_soxels()
entity_manager.sources[1].get_soxels()


from lib.fucntions import _generate_band_frequencies
from lib.acoustic_field import VelocityVectors, FrequencyLimitedField, AcousticField
from pympler import asizeof

low_freq = 5
high_freq = 48000 / 2
bands_per_octave = 24
num_bands_freq = len(_generate_band_frequencies(low_freq, high_freq, bands_per_octave))
vel = VelocityVectors(x=1.234, y=4.321, z=0.987)
flf = FrequencyLimitedField(low_freq=low_freq, high_freq=high_freq, pressure=0.123, velocity=vel)
acoustic_field = AcousticField()
for x in range(num_bands_freq):
    acoustic_field.add_field(low_freq=low_freq, high_freq=high_freq, pressure=0.123, velocity=vel)

# Get total size including referenced objects
total_size = asizeof.asizeof(acoustic_field)
print(f"Total size: {total_size} bytes")


def get_soxels():
    # Map type keywords to collections
type_map = [entity_manager.sources, entity_manager.objects, entity_manager.shaders]
soxel_list = []
for entities in type_map:
    for index in range(len(entities)):
        soxel_list += entities[index].get_soxels()

for index in range(len(soxel_list)):
    print(soxel_list[index][0])

    # Determine which collections to process
    if type is None:
        entities = type_map.values()
    else:
        type_lower = type.lower()
        entities = [coll for key, coll in type_map.items() if key in type_lower]

    # Process collections
    for entity in entities:
        if idx is not None:
            soxels = entity[idx].get_soxels()
        else:
            for element in entity.keys():
                soxels = entity[element].get_soxels()
    print(soxels)


from sources.spherical_source import SphericalSource
for source in config.sources:
    if source.idx == 0:
        source_config = source

source0 = SphericalSource(config, source_config)

center = [4,6,5]
point = [2,5,5]
sound_speed = 343.4 
density = 1.225
source_input = source0.get_field(current_frame, center, point, sound_speed, density)

ad_geometry = config.acoustic_domain.geometry
ad_geometry = ad_geometry if isinstance(ad_geometry, np.ndarray) else np.array(ad_geometry)
shape_z = (np.linalg.norm(ad_geometry[0] - ad_geometry[1] / config.acoustic_domain.voxel_size).astype(int))
shape_y = (np.linalg.norm(ad_geometry[1] - ad_geometry[2]) / config.acoustic_domain.voxel_size).astype(int)
shape_x = (np.linalg.norm(ad_geometry[2] - ad_geometry[6]) / config.acoustic_domain.voxel_size).astype(int)
config.acoustic_domain.shape = [int(shape_x), int(shape_y), int(shape_z)]

soxel_grid = SoxelGrid(config)

for obj in config.objects:
    if obj.idx == 0:
        config_obj = obj

config_obj.acoustic_shader.acoustic_properties.absorption.coefficients
freq, coeffs = config_obj.acoustic_shader.acoustic_properties.absorption.get_coeffs()
coeffs

config_obj.acoustic_shader.acoustic_properties.absorption.get_avg_coeffs()
config_obj.acoustic_shader.acoustic_properties.absorption.get_avg_coeffs(250,500)


for i in range(soxel_grid.shape[0]):
    for j in range(soxel_grid.shape[1]):
        for k in range(soxel_grid.shape[2]):
            if soxel_grid.soxels[i,j,k].type == 1:
                if soxel_grid.soxels[i,j,k].idx == 0:
                   pressure = soxel_grid.soxels[i,j,k].input_pressures.get_field(987.0149282610902, 1015.936673259656).pressure
                   velocity = soxel_grid.soxels[i,j,k].input_pressures.get_field(987.0149282610902, 1015.936673259656).velocity

config = em.get('config')
soxel_grid = em.get('soxel_grid')
for i in range(soxel_grid.shape[0]):
    for j in range(soxel_grid.shape[1]):
        for k in range(soxel_grid.shape[2]):
            if soxel_grid.soxels[i,j,k].type == 2:
                print(i,j,k)

for i in range(soxel_grid.shape[0]):
    for j in range(soxel_grid.shape[1]):
        for k in range(soxel_grid.shape[2]):
            if soxel_grid.soxels[i,j,k].type == 1:
                print(i,j,k)
                input_press = soxel_grid.soxels[i,j,k].input_pressures

low_freq=522.8529450371557, high_freq=538.1737057623811
soxel_grid.soxels[55,99,74].input_pressures

for index in range(len(input_press.field)):
    if freq == input_press.field[index].low_freq:
        print(index)

for i in range(soxel_grid.shape[0]):
    for j in range(soxel_grid.shape[1]):
        for k in range(soxel_grid.shape[2]):
            if soxel_grid.soxels[i,j,k].type == 1:
                for coord, soxel in soxel_list:
                    soxel_grid.soxels[coord] = soxel

config = em.get('config')
for source_config in config.sources:
    source = em.get('sources', source_config.idx)
    soxel_list = source.get_soxels()
    for i,j,k, soxel in soxel_list:
        print(i,j,k, soxel)
        soxel_grid.soxels[coord] = soxel

                print(i,j,k)
            if not soxel_grid.soxels[i,j,k].type == 0:
                print(i,j,k)
                soxel_grid.soxels[i,j,k]

for source in config.sources:
    if 'planar' in source.type:
        config_planar_source = source
    elif source.idx == 0:
        config_spherical_source = source

frequencies, responses = config_spherical_source.spatial_freq_response.get_responses()
frequencies
responses

frequencies, phases = config_spherical_source.spatial_freq_response.get_phases()
frequencies
phases

config_spherical_source.spatial_freq_response.get_avg_response()
config_spherical_source.spatial_freq_response.get_avg_phase()

frequencies, responses = config_spherical_source.spatial_freq_response.get_responses(azimuth=56, elevation=20, low_freq=34, high_freq=35)
frequencies
responses

frequencies, phases = config_spherical_source.spatial_freq_response.get_phases(azimuth=56, elevation=20, low_freq=34, high_freq=35)
frequencies
phases

config_spherical_source.spatial_freq_response.get_avg_response(azimuth=56, elevation=20, low_freq=34, high_freq=35)
config_spherical_source.spatial_freq_response.get_avg_phase(azimuth=56, elevation=20, low_freq=34, high_freq=35)

config_planar_source.spatial_freq_response.get_responses(98,67)
config_planar_source.spatial_freq_response.get_phases(98,67)
config_planar_source.spatial_freq_response.get_avg_response(23,34,23,100)
config_planar_source.spatial_freq_response.get_avg_phase(23,34,23,100)


center = []
if source_config.position_file and os.path.exists(source_config.position_file):
    centers = np.load(source_config.position_file)
    center = centers[centers.files[0]]
    center = np.array(centers[centers.files[0]])

def _world_to_grid(world_pos: Union[list, tuple, np.ndarray]) -> Tuple[int, int, int]:
    """Convert world coordinates to grid indices"""
    ad_geometry = config.acoustic_domain.geometry
    ad_geometry = ad_geometry if isinstance(ad_geometry, np.ndarray) else np.array(ad_geometry)
    world_pos = world_pos if isinstance(world_pos, np.ndarray) else np.array(world_pos)
    grid_coords = ((world_pos - ad_geometry[0]) / config.acoustic_domain.voxel_size).astype(int)
    return grid_coords

def _is_in_bounds(i: int, j: int, k: int) -> bool:
    """Check if grid indices are within bounds""" 
    return (0 <= i < config.acoustic_domain.shape[0] and
            0 <= j < config.acoustic_domain.shape[1] and 
            0 <= k < config.acoustic_domain.shape[2])

voxels = []
for vertex in vertices:
    i,j,k = _world_to_grid(vertex)
    if _is_in_bounds(i,j,k):
        voxels.append((i,j,k))

voxels


ad_geometry = config.acoustic_domain.geometry
voxelized_center = _world_to_grid(center[0])
radius = source_config.geometry
radius_sq = radius ** 2
min_x, min_y, min_z = _world_to_grid((center-radius))
max_x, max_y, max_z = _world_to_grid((center+radius))
voxels = []

for x in range(min_x, max_x + 1):
    for y in range(min_y, max_y + 1):
        for z in range(min_z, max_z + 1):
            dist = np.linalg.norm(voxelized_center - (x,y,z))
            if not dist > 0:
                voxels.append((x, y, z))
                print(x,y,z)


import soundfile as sf
source_config = config_source
audio_file = source_config.audio_file
audio_data, sample_rate = sf.read(audio_file)

npz_path = os.path.join(config.system.cache_path, 'filtered_audio')
audio_npz = str(source_config.idx) + '.npz'
npz_path = os.path.join(npz_path, audio_npz)

from lib.filter import LinkwitzRileyFilter
start_freq = 20
end_freq = 20000
frequencies = []
multi_bands_audio = []
current_freq = start_freq
step_ratio = 2 ** (1 / 24)
while current_freq <= end_freq:
    frequencies.append(current_freq)
    current_freq *= step_ratio

for index in range(len(frequencies)-1):
    low_freq = frequencies[index]
    high_freq = frequencies[index + 1]
    filtered_audio, sample_rate = LinkwitzRileyFilter.linkwitz_riley_bandpass_filter(audio_data, sample_rate, low_freq, high_freq)
    if not len(filtered_audio) == len(audio_data):
        print(f"Error filtered audio lenght {len(filtered_audio)} differ from audio data {len(audio_data)}")
        print(f"low frequency {low_freq}, high frequency {high_freq}")
    else:
        multi_bands_audio.append([low_freq, high_freq, filtered_audio])

############################

    def _get_sphere_surface_area(self, radius: float)
        """
        Calculate the surface area of a sphere defined by radius.

        Parameters:
        radius (float): the radius of the sphere

        Returns:
        float: Surface area of the sphere
        """
        return 4 * math.pi * radius ** 2

    def _get_planar_surface_area(self, vertices: np.array)
        """
        Calculate the surface area of a planar defined by vertices.
        The vertices should form a polygon (triangle, quadrilateral, etc.).

        Parameters:
        vertices (numpy.ndarray): Array of shape (n, 3) containing 3D coordinates of vertices

        Returns:
        float: Surface area of the planar
        """
        if not isinstance(vertices, np.ndarray):
            vertices = np.array(vertices)

        if vertices.shape[1] != 3:
            raise ValueError("Vertices must be 3D coordinates (shape: n x 3)")

        if vertices.shape[0] < 3:
            raise ValueError("At least 3 vertices are required to define a planar")

        # For triangles, use the cross product method
        if vertices.shape[0] == 3:
            v1 = vertices[1] - vertices[0]
            v2 = vertices[2] - vertices[0]
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))

        # For polygons with more than 3 vertices, triangulate and sum areas
        else:
            # Use the shoelace formula for 3D polygons by projecting to 2D
            # First, find the normal vector to the planar
            v1 = vertices[1] - vertices[0]
            v2 = vertices[2] - vertices[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)

            # Project vertices to 2D by removing the dimension with largest normal component
            max_idx = np.argmax(np.abs(normal))
            indices = [i for i in range(3) if i != max_idx]
            projected_vertices = vertices[:, indices]

            # Apply shoelace formula
            x = projected_vertices[:, 0]
            y = projected_vertices[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area
