import time
import numpy as np
import trimesh
import os, sys

from typing import Dict, List, Tuple

from embreex import rtcore_scene as rtcs
from embreex.mesh_construction import TriangleMesh

from pbrAudioRay.core.entity_manager import EntityManager
from pbrAudioRay.core.acoustic_engine import AcousticEngine

from pbrAudioRay.lib.frequency_bands import FrequencyBands

np.set_printoptions(precision=18, floatmode='fixed', threshold=np.inf)

def loop(origins, directions, energies, phases, mesh_info, scene_info, recursion_idx, ad_alpha, ad_beta, medium_info_alpha, medium_info_beta, abs_coeffs_info, abs_phases_info, refl_coeffs_info, refl_phases_info, refr_coeffs_info, refr_phases_info, scat_coeffs_info, scat_phases_info, roughness_info, frequency_bands):
    n_bands = len(frequency_bands)
    t1 = time.time()
    res = scene.run(origins, directions, output=1)
    t2 = time.time()
    print("Ran in {0:.3f} s".format(t2 - t1))
    ray_inter = res["geomID"] >= 0
    print("{0} rays intersect geometry (over {1})".format(sum(ray_inter), n_rays))
    primID = res["primID"][ray_inter]
    u = res["u"][ray_inter]
    v = res["v"][ray_inter]
    w = 1 - u - v
    a = mesh_info[primID][:, 0, :]
    b = mesh_info[primID][:, 1, :]
    c = mesh_info[primID][:, 2, :]
    inters = (np.vstack(w) * a + np.vstack(u) * b + np.vstack(v) * c)

    # Save ray_data origins and hit_points to json for analysis
    import json
    data_dict = {}
    data_dict['origins'] = origins.tolist()
    data_dict['hit_points'] = inters.tolist()
    filepath = f"ray_datas/embreex_{recursion_idx:04}.json"
    with open(filepath, 'w') as f:
        json.dump(data_dict, f, indent=2)

    # Compute traveled path length
    path_length = np.sqrt(np.sum((inters - origins)**2, axis=1)).reshape(-1,1)

    # Get intersect obj_idx
    hits_obj_idx = scene_info[primID]
    intersect_mask = hits_obj_idx >= 0
    intersect_obj_idx = hits_obj_idx[intersect_mask]

    # Get material property of intersected objects
    abs_coeffs = abs_coeffs_info[primID][intersect_mask]
    abs_phases = abs_phases_info[primID][intersect_mask]
    refl_coeffs = refl_coeffs_info[primID][intersect_mask]
    refl_phases = refl_phases_info[primID][intersect_mask]
    refr_coeffs = refr_coeffs_info[primID][intersect_mask]
    refr_phases = refr_phases_info[primID][intersect_mask]
    scat_coeffs = scat_coeffs_info[primID][intersect_mask]
    scat_phases = scat_phases_info[primID][intersect_mask]
    roughness_factor = roughness_info[primID][intersect_mask]

    # Init medium array
    sound_speed = config.acoustic_domain.acoustic_shader.sound_speed
    medium_speed = np.full((origins.shape[0],1), [sound_speed], dtype=np.float32)
    medium_alpha = np.full((origins.shape[0],n_bands), ad_alpha, dtype=np.float32)
    medium_beta = np.full((origins.shape[0],n_bands), ad_beta, dtype=np.float32)

    # Find medium object
    for obj_config in config.objects:
        mesh_file = f"{obj_config.obj_path}/{obj_config.name}.npz"
        if os.path.isfile(mesh_file):
#            print(mesh_file)
            data = np.load(mesh_file, allow_pickle=False)
            vertices = data[data.files[0]].astype(np.float32)
            vertex_normals = data[data.files[1]].astype(np.float32)
            faces = data[data.files[2]].astype(np.int32)
            mesh = trimesh.Trimesh(vertices=vertices, vertex_normals=vertex_normals, faces=faces)
            medium_mask = mesh.contains(origins)
            if np.any(medium_mask):
                # Get medium properties
                sound_speed = obj_config.acoustic_shader.sound_speed
                density = obj_config.acoustic_shader.density
                young_modulus = obj_config.acoustic_shader.young_modulus
                poisson_ratio = obj_config.acoustic_shader.poisson_ratio
                damping = obj_config.acoustic_shader.damping
                alpha, beta = _compute_acoustic_object_coefficients(sound_speed, density, young_modulus, poisson_ratio, damping)
                medium_speed[medium_mask] = sound_speed
                medium_alpha[medium_mask] = alpha
                medium_beta[medium_mask] = beta

    # Compute medium attenuation
    attenuation = np.exp(-medium_alpha * path_length)
    energies = energies * attenuation

    # Compute phase shift
    phase_shift = medium_beta * path_length
    phases = (phases + phase_shift) % (2 * np.pi)

    # Compute delay
    delay = path_length * medium_speed

    # Compute triangle normals using np.cross with broadcasting
    normals = np.cross(b-a, c-a)

    # Normalize (avoid division by zero)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # Filter directions, normals, energies, phases
    normals = normals[intersect_mask]
    directions = directions[ray_inter][intersect_mask]
    energies = energies[ray_inter][intersect_mask]
    phases = phases[ray_inter][intersect_mask]

    # Compute reflection directions
    dot = np.sum(directions * normals, axis=1)
    incident_angles = np.arccos(-dot)
    reflected_directions = directions - 2 * dot[:, np.newaxis] * normals
    recursion_idx += 1
    reflected_directions = reflected_directions.astype(np.float32)

    # Compute absorption
    angle_factor = np.cos(incident_angles)
    angle_factor[angle_factor == 0] = 1e-10
    absorbed_energies = energies * angle_factor.reshape(-1,1) * abs_coeffs.reshape(-1,1)
#    print('absorbed_energies', absorbed_energies)

    # Compute reflection absorption
    reflected_energies = energies * refl_coeffs.reshape(-1,1)
    reflected_phase = phases + refl_phases.reshape(-1,1) % (2 * np.pi)
#    print('reflected_energies', reflected_energies, reflected_phase)

    # Compute scattering absorption
    scattered_directions = _random_hemisphere_directions(normals)
    scattered_energies = energies * scat_coeffs * roughness_factor / max(scattered_directions.shape[0], 1e-10)
    scattered_phase = phases + scat_phases.reshape(-1,1) % (2 * np.pi)
#    print('scattered_energies', scattered_energies, scattered_phase)

    # Energy conservation check
    total_out = energies + absorbed_energies + reflected_energies + scattered_energies
    delta_energies = abs(total_out - energies)
    delta_mask = delta_energies > 1e-10
    if not np.all(delta_mask):
        total_out[~delta_mask] = energies[~delta_mask] + 1e-10

    # Normalize to ensure energies conservation
    scale = energies / total_out
    absorbed_energies *= scale
    reflected_energies *= scale
    scattered_energies *= scale

    # Detach new origins from triangles surface along normals of 0.001 factor
    origins = inters[intersect_mask] + 0.001 * normals
    origins = origins.astype(np.float32)

    # Create new origins, directions, energies, phases
    origins = np.append(origins, origins, axis=0).astype(np.float32)
    directions = np.append(reflected_directions, scattered_directions, axis=0).astype(np.float32)
    energies = np.append(reflected_energies, scattered_energies, axis=0).astype(np.float32)
    phases = np.append(reflected_phase, scattered_phase, axis=0).astype(np.float32)

    # Filter direction, origins and phases on energy termination
    termination_energy = 1e-64 # config.termination.energy_threshold
    termination_mask = energies > termination_energy
    energies = energies[termination_mask].reshape(-1,1)
    phases = phases[termination_mask].reshape(-1,1)
    directions = directions[termination_mask.reshape(-1,)]
    origins = origins[termination_mask.reshape(-1,)]

    print('termination', np.count_nonzero(termination_mask))

    loop(origins, directions, energies, phases, mesh_info, scene_info, recursion_idx, ad_alpha, ad_beta, medium_info_alpha, medium_info_beta, abs_coeffs_info, abs_phases_info, refl_coeffs_info, refl_phases_info, refr_coeffs_info, refr_phases_info, scat_coeffs_info, scat_phases_info, roughness_info, frequency_bands)

def _random_hemisphere_directions(normals: np.ndarray) -> np.ndarray:
    """
    Generate random directions on hemispheres oriented along the normals.
    Args:
        normals: Surface normals
        n_samples: Number of samples to generate
    Returns:
        Array of sampled directions
    """
    n_samples = max(normals.shape[0], 1)
    directions = np.random.uniform(-1,1,(n_samples,3))
    directions /= np.linalg.norm(directions)
    while not np.all(directions[:, 0]**2 + directions[:, 1]**2 + directions[:, 2]**2 < 1):
        directions = np.random.uniform(-1,1,(n_samples,3))
        # Project onto hemisphere oriented along normals
        directions /= np.linalg.norm(directions)
    # Flip if pointing away from normals
    if not np.any(np.sum(directions * normals, axis=1) < 0):
        mask = np.sum(directions * normals, axis=1) < 0
        directions[mask] = -directions[mask]
    return directions

def _compute_acoustic_object_coefficients(c: float, rho: float, E: float, nu: float, damping: float, freq_bands: List[Tuple[float,float]]):
    """
    Calculate medium attenuation coefficient and phase shift for acoustic objects.
    Works for gases, fluids, and solids using a simplified common method.
    """
    # Compute Rayleigh damping coefficient α (mass proportional) and Rayleigh damping coefficient β (stiffness proportional)
    n_bands = len(freq_bands)
    alpha = beta = np.zeros((n_bands,1), dtype=np.float32)
    for idx in range(n_bands):
        min_freq, max_freq = freq_bands[idx]
        alpha[idx], beta[idx] = _compute_rayleigh_damping(min_freq, max_freq, damping)
    freqs = np.unique(freq_bands)[:-1]
    omega = 2 * np.pi * freqs
    omega = omega.reshape(freqs.shape[0],1)
    # Calculate derived properties from input parameters.
    # Bulk modulus (for fluids and isotropic solids)
    K = E / (3 * (1 - 2 * nu))
    # Shear modulus (for solids)
    G = E / (2 * (1 + nu))
    # Characteristic impedance
    Z = rho * c
    # Determine if it's likely a solid (has significant shear modulus)
    is_solid = G > 0.1 * E
    # Compute attenuation coefficient α_att (in Np/m) using Rayleigh damping model: α_att = (α / (2*c)) + (β * ω² / (2*c))
    alpha_attenuation = (alpha / (2 * c)) + (beta * omega**2 / (2 * c))
    # For fluids and gases add a simple viscous term
    if not is_solid:
        # Simplified viscous attenuation (Stokes' law approximation)
        # Assuming dynamic viscosity ≈ 1.8e-5 Pa·s for air, 1e-3 Pa·s for water
        viscosity = 1.8e-5 if rho < 100 else 1e-3  # rough approximation
        alpha_viscous = (2 * omega**2 * viscosity) / (3 * rho * c**3)
    else:
        alpha_viscous = 0
    # Total attenuation coefficient
    alpha_attenuation = alpha_attenuation + alpha_viscous
    # Compute phase shift (in rads/m)
    phase_shift = omega / c
    return alpha_attenuation, phase_shift

def _compute_acoustic_domain_coefficients(c: float, rho: float, T: float, Z: float, freq_bands: List[Tuple[float,float]]):
    """
    Compute absorption coefficient and phase shift coefficient for air.
    Parameters:
    -----------
    c : float
        Speed of sound in air (m/s)
    rho : float
        Density of of air (kg/m³)
    T : float
        Temperature in °C
    Z : float
        Characteristic impedance of air (rayls)
    freqs : array
        Frequency (Hz) - bands frequency
    Returns:
    --------
    alpha : array
        Absorption coefficient (nepers/m)
    beta : array
        Phase shift coefficient (rad/m)
    """
    freqs = np.unique(freq_bands)[:-1]
    # Convert temperature to Kelvin
    T_K = T + 273.15
    # Compute Angular frequency
    omega = 2 * np.pi * freqs
    # Stokes-Kirchhoff formula for sound absorption
    # Constants for air
    mu = 1.846e-5  # Dynamic viscosity (Pa·s) at 20°C
    kappa = 0.0262  # Thermal conductivity (W/m·K) at 20°C
    Cp = 1005  # Specific heat at constant pressure (J/kg·K)
    Cv = 718  # Specific heat at constant volume (J/kg·K)
    gamma_specific = Cp / Cv  # Ratio of specific heats
    # Viscous contribution
    alpha_visc = (omega**2 * mu) / (2 * rho * c**3)
    # Thermal contribution
    alpha_therm = (omega**2 * kappa * (gamma_specific - 1)) / (2 * rho * c**3 * Cp)
    # Total absorption coefficient
    alpha = alpha_visc + alpha_therm
    # Phase shift coefficient
    beta = omega / c
    return alpha, beta


config_file = 'config.json'
entity_manager = EntityManager(config_file)

config = entity_manager.get('config')

frequency_bands = FrequencyBands(entity_manager)
n_bands = len(frequency_bands.get_bands())
n_objs = len(config.objects)

recursion_idx = 0
mesh_info = np.zeros((0,3,3), dtype=np.float32)
scene_info = np.zeros((0,1), dtype=np.float32)
medium_info_alpha = np.zeros((n_objs,n_bands), dtype=np.float32)
medium_info_beta = np.zeros((n_objs,n_bands), dtype=np.float32)
roughness_info = np.zeros((0,1), dtype=np.float32)
abs_coeffs_info = np.zeros((0,n_bands), dtype=np.float32)
abs_phases_info = np.zeros((0,n_bands), dtype=np.float32)
refl_coeffs_info = np.zeros((0,n_bands), dtype=np.float32)
refl_phases_info = np.zeros((0,n_bands), dtype=np.float32)
refr_coeffs_info = np.zeros((0,n_bands), dtype=np.float32)
refr_phases_info = np.zeros((0,n_bands), dtype=np.float32)
scat_coeffs_info = np.zeros((0,n_bands), dtype=np.float32)
scat_phases_info = np.zeros((0,n_bands), dtype=np.float32)

# Init embree scene
scene = rtcs.EmbreeScene()

# Acoustic Domain mesh
ac_geometry = np.array(config.acoustic_domain.geometry)
ac_max = np.array([max(ac_geometry[i][0] for i in range(len(ac_geometry))), max(ac_geometry[i][1] for i in range(len(ac_geometry))), max(ac_geometry[i][2] for i in range(len(ac_geometry)))])
ac_min = np.array([min(ac_geometry[i][0] for i in range(len(ac_geometry))), min(ac_geometry[i][1] for i in range(len(ac_geometry))), min(ac_geometry[i][2] for i in range(len(ac_geometry)))])
mesh = trimesh.creation.box(bounds=(ac_min,ac_max))

vertices = mesh.vertices.astype(np.float32)
faces = mesh.faces.astype(np.int32)
mesh_info = np.append(mesh_info, mesh.vertices[mesh.faces], axis=0)
scene_info = np.append(scene_info, np.full((mesh.vertices[mesh.faces].shape[0],), [-1], dtype=np.int32))

# Compute default medium properties
sound_speed = config.acoustic_domain.acoustic_shader.sound_speed
density = config.acoustic_domain.acoustic_shader.density
temperature = config.acoustic_domain.acoustic_shader.temperature
impedence = config.acoustic_domain.acoustic_shader.impedence
ad_alpha, ad_beta = _compute_acoustic_domain_coefficients(sound_speed, density, temperature, impedence, frequency_bands.get_bands())

# Objects mesh
for obj_config in config.objects:
    mesh_file = f"{obj_config.obj_path}/{obj_config.name}.npz"
    if os.path.isfile(mesh_file):
#        print(mesh_file)
        data = np.load(mesh_file, allow_pickle=False)
        vertices = data[data.files[0]].astype(np.float32)
        faces = data[data.files[2]].astype(np.int32)
        mesh_info = np.append(mesh_info, vertices[faces], axis=0)
        scene_info = np.append(scene_info, np.full((vertices[faces].shape[0],), [obj_config.idx], dtype=np.int32))
        # Get object shader properties
        roughness = obj_config.acoustic_shader.roughness
        roughness_info = np.append(roughness_info, np.full((vertices[faces].shape[0],1), [roughness], dtype=np.float32), axis=0)
        # Get material properties absorption
        abs_coeffs, abs_phases = obj_config.acoustic_shader.acoustic_properties.absorption.get_bands_avg(frequency_bands.get_bands())
        abs_coeffs_info = np.append(abs_coeffs_info, np.full((vertices[faces].shape[0],n_bands), abs_coeffs, dtype=np.float32), axis=0)
        abs_phases_info = np.append(abs_phases_info, np.full((vertices[faces].shape[0],n_bands), abs_phases, dtype=np.float32), axis=0)
        # Get material properties reflecrtion
        refl_coeffs, refl_phases = obj_config.acoustic_shader.acoustic_properties.reflection.get_bands_avg(frequency_bands.get_bands())
        refl_coeffs_info = np.append(refl_coeffs_info, np.full((vertices[faces].shape[0],n_bands), refl_coeffs, dtype=np.float32), axis=0)
        refl_phases_info = np.append(refl_phases_info, np.full((vertices[faces].shape[0],n_bands), refl_phases, dtype=np.float32), axis=0)
        # Get material properties refraction
        refr_coeffs, refr_phases = obj_config.acoustic_shader.acoustic_properties.refraction.get_bands_avg(frequency_bands.get_bands())
        refr_coeffs_info = np.append(refr_coeffs_info, np.full((vertices[faces].shape[0],n_bands), refr_coeffs, dtype=np.float32), axis=0)
        refr_phases_info = np.append(refr_phases_info, np.full((vertices[faces].shape[0],n_bands), refr_phases, dtype=np.float32), axis=0)
        # Get material properties refraction
        scat_coeffs, scat_phases = obj_config.acoustic_shader.acoustic_properties.scattering.get_bands_avg(frequency_bands.get_bands())
        scat_coeffs_info = np.append(scat_coeffs_info, np.full((vertices[faces].shape[0],n_bands), scat_coeffs, dtype=np.float32), axis=0)
        scat_phases_info = np.append(scat_phases_info, np.full((vertices[faces].shape[0],n_bands), scat_phases, dtype=np.float32), axis=0)

embree_mesh = TriangleMesh(scene, mesh_info)

pose = np.load('data/pose/Camera.npz')
output_pos = pose[pose.files[0]].reshape(-1,3)

pose = np.load('data/pose/SphericalSource.npz')
source_pos = pose[pose.files[0]].reshape(-1,3)

n_rays = config.system.number_of_rays
n_src = source_pos.shape[0]

main_dirs = output_pos - source_pos
main_dirs_norm = np.linalg.norm(main_dirs, axis=1, keepdims=True)
main_dirs_norm[main_dirs_norm <= 1e-10] = 1e-10
main_dirs = main_dirs / main_dirs_norm

num_points = n_rays * n_src

phi = np.pi * (3. - np.sqrt(5.))
theta = phi * np.arange(num_points)
z = np.linspace(1/num_points-1, 1-1/num_points, num_points)
radius = np.sqrt(1 - z * z)
y = radius * np.sin(theta)
x = radius * np.cos(theta)

directions = np.array(list(zip(x,y,z)), dtype=np.float32)

for idx in range(n_src):
    index = n_rays * idx
    directions[index] = main_dirs[idx]

origins = np.zeros((0,3), dtype=np.float32)
for idx in range(n_src):
    source_arr = np.full((n_rays,3), source_pos[idx], dtype=np.float32)
    origins = np.append(origins, source_arr, axis=0)

energies = np.full((num_points,1), [1], dtype=np.float32)
phases = np.full((num_points,1), [0], dtype=np.float32)

loop(origins, directions, energies, phases, mesh_info, scene_info, recursion_idx, ad_alpha, ad_beta, medium_info_alpha, medium_info_beta, abs_coeffs_info, abs_phases_info, refl_coeffs_info, refl_phases_info, refr_coeffs_info, refr_phases_info, scat_coeffs_info, scat_phases_info, roughness_info, frequency_bands.get_bands())

