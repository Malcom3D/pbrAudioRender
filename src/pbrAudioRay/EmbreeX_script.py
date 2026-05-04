#EmbreeX_script-0.0.13
import time
import numpy as np
import trimesh
import os, sys

import resampy
import soundfile as sf
from typing import Dict, List, Tuple

from embreex import rtcore_scene as rtcs
from embreex.mesh_construction import TriangleMesh

from scipy.signal import fftconvolve, convolve

from pbrAudioRay.core.entity_manager import EntityManager
from pbrAudioRay.core.acoustic_engine import AcousticEngine

from pbrAudioRay.lib.frequency_bands import FrequencyBands
from pbrAudioRay.lib.functions import _compute_rayleigh_damping

np.set_printoptions(precision=18, floatmode='fixed', threshold=np.inf)

def loop(origins, origins_idx, origins_bands, destinations, destinations_idx, directions, energies, phases, delay, mesh_info, scene_info, recursion_idx, ad_alpha, ad_beta, medium_info_alpha, medium_info_beta, abs_coeffs_info, abs_phases_info, refl_coeffs_info, refl_phases_info, refr_coeffs_info, refr_phases_info, scat_coeffs_info, scat_phases_info, roughness_info, frequency_bands, output_source, output_bands, output_energies, output_phases, output_delay, output_origins, output_directions, output_destinations, output_destinations_idx):
    n_bands = len(frequency_bands)
    t1 = time.time()
    res = scene.run(origins, directions, output=1)
    t2 = time.time()
    print("Ran in {0:.3f} s".format(t2 - t1))
    ray_inter = res["geomID"] >= 0
    print("{0} rays intersect geometry (over {1})".format(sum(ray_inter), origins.shape[0]))
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

    # Filter origins, origins_idx, origins_bands, destinations, destinations_idx and directions
    origins = origins[ray_inter]
    origins_idx = origins_idx[ray_inter]
    origins_bands = origins_bands[ray_inter]
    destinations = destinations[ray_inter]
    destinations_idx = destinations_idx[ray_inter]
    directions = directions[ray_inter]

    # Compute traveled path length
    path_length = np.sqrt(np.sum((inters - origins)**2, axis=1)).reshape(-1,1)

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
                print('Find medium object properties for', obj_config.name)
                # Get medium properties
                sound_speed = obj_config.acoustic_shader.sound_speed
                density = obj_config.acoustic_shader.density
                young_modulus = obj_config.acoustic_shader.young_modulus
                poisson_ratio = obj_config.acoustic_shader.poisson_ratio
                damping = obj_config.acoustic_shader.damping
                alpha, beta = _compute_acoustic_object_coefficients(sound_speed, density, young_modulus, poisson_ratio, damping, frequency_bands)
                medium_speed[medium_mask] = sound_speed
                medium_alpha[medium_mask] = alpha.T
                medium_beta[medium_mask] = beta.T

    # Filter energies, phases and delay
    energies = energies[ray_inter]
    phases = phases[ray_inter]
    delay = delay[ray_inter]

    # Compute medium attenuation
    origins_bands_idx = np.arange(origins_bands.T.shape[0])
    attenuation = np.exp(-medium_alpha * path_length)
    energies = energies * attenuation[origins_bands_idx,origins_bands]

    # Compute phase shift
    phase_shift = path_length * medium_beta[origins_bands_idx,origins_bands]
    phases = (phases + phase_shift) % (2 * np.pi)

    # Compute delay
    new_delay = path_length / medium_speed
    delay = delay + new_delay

    # Get intersect obj_idx
    hits_obj_idx = scene_info[primID]
    intersect_mask = hits_obj_idx >= 0
    intersect_obj_idx = hits_obj_idx[intersect_mask]

    # Get output data
    output_mask = hits_obj_idx <= -3
    output_source = np.append(output_source, origins_idx[output_mask], axis=0).astype(np.int32)
    output_bands = np.append(output_bands, origins_bands[output_mask], axis=0).astype(np.int32)
    output_energies = np.append(output_energies, energies[output_mask], axis=0).astype(np.float32)
    output_phases = np.append(output_phases, phases[output_mask], axis=0).astype(np.float32)
    output_delay = np.append(output_delay, delay[output_mask], axis=0).astype(np.float32)
    output_origins = np.append(output_origins, origins[output_mask], axis=0).astype(np.float32)
    output_destinations = np.append(output_destinations, destinations[output_mask], axis=0).astype(np.float32)
    output_destinations_idx = np.append(output_destinations_idx, destinations_idx[output_mask], axis=0).astype(np.float32)
    output_directions = np.append(output_directions, directions[output_mask], axis=0).astype(np.float32)
    
    print('output:', np.count_nonzero(output_mask), output_energies.shape[0])

    # filter inters coordinate, path_length and delay
    inters = inters[intersect_mask]
    path_length = path_length[intersect_mask]
    delay = delay[intersect_mask]

    # Compute triangle normals using np.cross with broadcasting
    normals = np.cross(b-a, c-a)

    # Normalize (avoid division by zero)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # Filter medium_speed
    medium_speed = medium_speed[intersect_mask]
    # Filter directions, normals, energies, phases
    origins = origins[intersect_mask]
    origins_idx = origins_idx[intersect_mask]
    origins_bands = origins_bands[intersect_mask]
    destinations = destinations[intersect_mask]
    destinations_idx = destinations_idx[intersect_mask]
    normals = normals[intersect_mask]
#    directions = directions[ray_inter][intersect_mask]
    directions = directions[intersect_mask]
    energies = energies[intersect_mask]
    phases = phases[intersect_mask]

    # Get material property of intersected objects
    origins_bands_idx = np.arange(origins_bands.T.shape[0])
    abs_coeffs = abs_coeffs_info[primID][intersect_mask][origins_bands_idx,origins_bands]
    abs_phases = abs_phases_info[primID][intersect_mask][origins_bands_idx,origins_bands]
    refl_coeffs = refl_coeffs_info[primID][intersect_mask][origins_bands_idx,origins_bands]
    refl_phases = refl_phases_info[primID][intersect_mask][origins_bands_idx,origins_bands]
    refr_coeffs = refr_coeffs_info[primID][intersect_mask][origins_bands_idx,origins_bands]
    refr_phases = refr_phases_info[primID][intersect_mask][origins_bands_idx,origins_bands]
    scat_coeffs = scat_coeffs_info[primID][intersect_mask][origins_bands_idx,origins_bands]
    scat_phases = scat_phases_info[primID][intersect_mask][origins_bands_idx,origins_bands]
    roughness_factor = roughness_info[primID][intersect_mask]

    # Compute absorption
    dot_projection = np.sum(directions * normals, axis=1)
    incident_angles = np.arccos(-dot_projection)
    angle_factor = np.cos(incident_angles)
    angle_factor[angle_factor == 0] = 1e-16
    absorbed_energies = energies * angle_factor.reshape(-1,1) * abs_coeffs.reshape(-1,1)

    # Compute reflection absorption
    reflected_energies = energies * refl_coeffs.reshape(-1,1)
    reflected_phases = - (phases + refl_phases.reshape(-1,1)) % (2 * np.pi)
    #print('reflected_energies', np.max(reflected_energies), np.min(reflected_energies))

    # Compute scattering absorption
    max_scattering = config.interface.max_scattering
    scattered_energies = energies * scat_coeffs * roughness_factor
    scattered_phases = phases + scat_phases.reshape(-1,1) % (2 * np.pi)
    #print('scattered_energies', np.max(scattered_energies), np.min(scattered_phases))

    # Energy conservation check
    total_out = energies + absorbed_energies + reflected_energies + scattered_energies
    delta_energies = abs(total_out - energies)
    delta_mask = delta_energies > 1e-16
    if not np.all(delta_mask):
        total_out[~delta_mask] = energies[~delta_mask] + 1e-16

    # Normalize to ensure energies conservation
    scale = energies / total_out
    absorbed_energies *= scale
    reflected_energies *= scale
    scattered_energies *= scale
    #print('scattered_energies', np.max(scattered_energies), np.min(scattered_phases))

    # Detach new origins from triangles surface along normals of 0.001 factor
    origins = inters + (0.01 * normals)
    origins = origins.astype(np.float32)

    # Compute wavelength
    frequency_mean = np.mean(frequency_bands, axis=1)
    wavelength = medium_speed.T / frequency_mean[origins_bands.T]
    # Compute Rayleigh parameter
    g = (2 * np.pi * roughness_factor * angle_factor.reshape(-1,1) / wavelength.T) ** 2

    # Specular reflection reduction factor
    reduction_factor = np.exp(-g)
    # Scattered energy creates a deviation in the effective reflection angle
    scattering_angles = np.arctan2(np.sin(incident_angles.reshape(-1,1)), np.cos(incident_angles.reshape(-1,1)) * (1 + g / 2))
    # Combine specular and scattered components
    reflection_angles = incident_angles.reshape(-1,1) * (1 - reduction_factor) + scattering_angles * reduction_factor

    # Compute reflection directions
    reflected_directions = _compute_reflection_directions(directions, normals, reflection_angles)
    reflected_directions = reflected_directions.astype(np.float32)

    # Compute scattering directions
    scattered_origins, scattered_origins_idx, scattered_origins_bands, scattered_destinations, scattered_destinations_idx, scattered_energies, scattered_phases, scattered_delay, scattered_directions = _random_hemisphere_directions(origins, origins_idx, origins_bands, destinations, destinations_idx, scattered_energies, scattered_phases, delay, normals, roughness_factor, max_scattering)

    # Create new origins, origins_idx, origins_bands, directions, energies, phases
    origins = np.append(origins, scattered_origins, axis=0).astype(np.float32)
    origins_idx = np.append(origins_idx, scattered_origins_idx, axis=0).astype(np.int32)
    origins_bands = np.append(origins_bands, scattered_origins_bands, axis=0).astype(np.int32)
    destinations = np.append(destinations, scattered_destinations, axis=0).astype(np.float32)
    destinations_idx = np.append(destinations_idx, scattered_destinations_idx, axis=0).astype(np.float32)
    directions = np.append(reflected_directions, scattered_directions, axis=0).astype(np.float32)
    energies = np.append(reflected_energies, scattered_energies, axis=0).astype(np.float32)
    phases = np.append(reflected_phases, scattered_phases, axis=0).astype(np.float32)
    delay = np.append(delay, scattered_delay, axis=0).astype(np.float32)

    # Filter direction, origins, phases, energy and delay on termination
    termination_energy = 1e-16 # config.termination.energy_threshold
    termination_mask = energies > termination_energy
    destinations = destinations[termination_mask.reshape(-1,)]
    destinations_idx = destinations_idx[termination_mask].reshape(-1,1)
    directions = directions[termination_mask.reshape(-1,)]
    origins = origins[termination_mask.reshape(-1,)]
    origins_idx = origins_idx[termination_mask].reshape(-1,1)
    origins_bands = origins_bands[termination_mask].reshape(-1,1)
    energies = energies[termination_mask].reshape(-1,1)
    phases = phases[termination_mask].reshape(-1,1)
    delay = delay[termination_mask].reshape(-1,1)

    print('termination', np.count_nonzero(termination_mask < 1))

    recursion_idx += 1
    if origins.shape[0] == 0:
        # compute and save the ambisonic IR 
        ambisonics_ir = compute_and_save_ir(output_source, output_bands, output_energies, output_phases, output_delay, output_origins, output_destinations, output_destinations_idx, output_directions, frequency_bands)
        sample_rate = int(config.system.sample_rate)

#        for src_config in config.sources:
#            if hasattr(src_config, 'audio_file') and os.path.exists(src_config.audio_file):
#                file_audio = src_config.audio_file
#                if file_audio.endswith('.wav'):
#                    mono_audio, samplerate = sf.read(file_audio)
#                    if not samplerate == sample_rate:
#                        mono_audio = resampy.resample(mono_audio, samplerate, sample_rate)
#                elif file_audio.endswith('.raw'):
#                    mono_audio = np.fromfile(audio_file, dtype=np.float32)
#
#                # Convolve Mono Audio track with AmbisonicIR
#                ambisonics_output = convolve_mono_to_ambisonics(mono_audio, ambisonics_ir, method='direct')
#                ambisonics_output_normalized = normalize_ambisonics_output(ambisonics_output)
#
#                # Transpose to get (n_samples, n_channels)
#                ambisonics_output_normalized = ambisonics_output_normalized.T
#
#                # Save the ambisonics_output_normalized to multitrack WAV file
#                output_dir = 'ambisonics_output'
#                os.makedirs(output_dir, exist_ok=True)
#                subtype='FLOAT'
#                filename='ambisonics_output_normalized.wav'
#                sf.write(filename, ambisonics_output_normalized, sample_rate, subtype=subtype)
#
#                print(f"Saved multitrack WAV file: {filename}")
#                print(f"Shape: {ambisonics_output_normalized.shape} (samples, channels)")
#                print(f"Sample rate: {sample_rate} Hz")
#                print(f"Duration: {ambisonics_output_normalized.shape[0] / sample_rate:.2f} seconds")
#                print(f"Format: {subtype}")

        return

    loop(origins, origins_idx, origins_bands, destinations, destinations_idx, directions, energies, phases, delay, mesh_info, scene_info, recursion_idx, ad_alpha, ad_beta, medium_info_alpha, medium_info_beta, abs_coeffs_info, abs_phases_info, refl_coeffs_info, refl_phases_info, refr_coeffs_info, refr_phases_info, scat_coeffs_info, scat_phases_info, roughness_info, frequency_bands, output_source, output_bands, output_energies, output_phases, output_delay, output_origins, output_directions, output_destinations, output_destinations_idx)

def compute_and_save_ir(output_source, output_bands, output_energies, output_phases, output_delay, output_origins, output_destinations, output_destinations_idx, output_directions, freq_bands: List[Tuple[float,float]]):

    config = entity_manager.get('config')
    sample_rate = int(config.system.sample_rate)
    s_src = len(config.sources)
    n_outs = len(config.outputs)
    n_bands = len(freq_bands)

    # Sort output by delay
    delay, energies, phases, sources, bands, origins, destinations, destinations_idx, directions = zip(*(sorted(zip(output_delay.tolist(), output_energies.tolist(), output_phases.tolist(), output_source.tolist(), output_bands.tolist(), output_origins.tolist(), output_destinations.tolist(), output_destinations_idx.tolist(), output_directions.tolist()))))
    delay = np.array(delay)
    energies = np.array(energies)
    phases = np.array(phases)
    sources = np.array(sources)
    bands = np.array(bands)
    origins = np.array(origins)
    destinations = np.array(destinations)
    destinations_idx = np.array(destinations_idx)
    directions = np.array(directions)

    # Convert delay to samples
    delay_samples = np.round(delay * sample_rate).astype(int)

    # Determine max IR length
    ir_length = int(np.ceil(np.max(delay_samples))) + 10

#    print('ir_length', ir_length)

    # filter by src-out for bands_idx
    for src_idx in range(n_src):
        for out_idx in range(n_outs):
            for bands_idx in range(n_bands):
                # filter by bands_idx
                bands_mask = bands == bands_idx
                a_sources = sources[bands_mask]
                a_energies = energies[bands_mask]
                a_phases = phases[bands_mask]
                a_delay_samples = delay_samples[bands_mask]
                a_bands = bands[bands_mask]
                a_origins = origins[bands_mask.reshape(-1,)]
                a_destinations = destinations[bands_mask.reshape(-1,)]
                a_destinations_idx = destinations_idx[bands_mask]
                a_directions = directions[bands_mask.reshape(-1,)]

                # filter by scr_idx
                source_mask = a_sources == src_idx
                a_sources = a_sources[source_mask]
                a_energies = a_energies[source_mask]
                a_phases = a_phases[source_mask]
                a_delay_samples = a_delay_samples[source_mask]
                a_bands = a_bands[source_mask]
                a_origins = a_origins[source_mask]
                a_destinations = a_destinations[source_mask]
                a_destinations_idx = a_destinations_idx[source_mask]
                a_directions = a_directions[source_mask]

                # filter by out_idx
                destination_mask = a_destinations_idx == out_idx
                a_sources = a_sources[destination_mask]
                a_energies = a_energies[destination_mask]
                a_phases = a_phases[destination_mask]
                a_delay_samples = a_delay_samples[destination_mask]
                a_bands = a_bands[destination_mask]
                a_origins = a_origins[destination_mask]
                a_destinations = a_destinations[destination_mask]
                a_destinations_idx = a_destinations_idx[destination_mask]
                a_directions = a_directions[destination_mask]

                for out_config in config.outputs: 
                    if out_config.idx == out_idx:
                        ambisonic_order = out_config.order

                # Initialize IR buffer
                n_channels = (ambisonic_order + 1) ** 2
                ambisonic_ir = np.zeros((n_channels, ir_length), dtype=np.float32)

#                print('ambisonic_ir', ambisonic_ir.shape)

                # Compute complex amplitudes
                complex_amplitudes = np.sqrt(a_energies) * np.exp(1j * a_phases)

#                print('complex_amplitudes', complex_amplitudes.shape)

                # Convert Cartesian to spherical coordinates
                x, y, z = a_directions[:, 0], a_directions[:, 1], a_directions[:, 2]

                # Spherical coordinates: theta (azimuth), phi (elevation)
                theta = np.arctan2(y, x)  # Azimuth
                phi = np.arcsin(z)  # Elevation (assuming unit vectors)

#                print('theta', theta.shape)
#                print('phi', phi.shape)

                # Compute spherical harmonics coefficients
                if ambisonic_order >= 0:
                    # Order 0: W channel (omnidirectional)
                    Y_00 = 1.0 / np.sqrt(4 * np.pi)  # SN3D normalization
    
                    for i in range(len(a_delay_samples)):
                        sample_idx = a_delay_samples[i]
                        ambisonic_ir[0, sample_idx] += np.real(complex_amplitudes[i] * Y_00)
#                print('W channel (omnidirectional)', np.count_nonzero(ambisonic_ir[0]), np.max(ambisonic_ir[0]), np.min(ambisonic_ir[0]))

                if ambisonic_order >= 1:
                    # Order 1: X, Y, Z channels
                    # These are the ACN channel ordering: W, Y, Z, X (for FuMa) or W, X, Y, Z (for ACN)
                    # Using ACN ordering (most common in modern ambisonics)
                    Y_1n1 = np.sqrt(3/(4*np.pi)) * np.sin(theta) * np.cos(phi)  # Y channel
                    Y_10 = np.sqrt(3/(4*np.pi)) * np.sin(phi)                   # Z channel
                    Y_11 = np.sqrt(3/(4*np.pi)) * np.cos(theta) * np.cos(phi)   # X channel
    
                    for i in range(len(a_delay_samples)):
                        sample_idx = a_delay_samples[i]
                        ambisonic_ir[1, sample_idx] += np.real(complex_amplitudes[i] * Y_1n1[i])
                        ambisonic_ir[2, sample_idx] += np.real(complex_amplitudes[i] * Y_10[i])
                        ambisonic_ir[3, sample_idx] += np.real(complex_amplitudes[i] * Y_11[i])
#                    print('X channel (Order 1)', np.count_nonzero(ambisonic_ir[1]))
#                    print('Y channel (Order 1)', np.count_nonzero(ambisonic_ir[2]))
#                    print('Z channel (Order 1)', np.count_nonzero(ambisonic_ir[3]))

                if ambisonic_order >= 2:
                    # Order 2: Additional 5 channels
                    # Second order spherical harmonics
                    sin_theta = np.sin(theta)
                    cos_theta = np.cos(theta)
                    sin_phi = np.sin(phi)
                    cos_phi = np.cos(phi)
    
                    # Precompute common terms
                    sqrt_15_4pi = np.sqrt(15/(4*np.pi))
                    sqrt_5_16pi = np.sqrt(5/(16*np.pi))
    
                    Y_2n2 = sqrt_15_4pi * sin_theta * cos_theta * cos_phi**2  # R channel
                    Y_2n1 = sqrt_15_4pi * sin_theta * sin_phi * cos_phi       # S channel
                    Y_20 = np.sqrt(5/(16*np.pi)) * (3*sin_phi**2 - 1)          # T channel
                    Y_21 = sqrt_15_4pi * cos_theta * sin_phi * cos_phi         # U channel
                    Y_22 = sqrt_15_4pi * (cos_theta**2 - sin_theta**2) * cos_phi**2  # V channel
    
                    for i in range(len(a_delay_samples)):
                        sample_idx = a_delay_samples[i]
                        ambisonic_ir[4, sample_idx] += np.real(complex_amplitudes[i] * Y_2n2[i])
                        ambisonic_ir[5, sample_idx] += np.real(complex_amplitudes[i] * Y_2n1[i])
                        ambisonic_ir[6, sample_idx] += np.real(complex_amplitudes[i] * Y_20[i])
                        ambisonic_ir[7, sample_idx] += np.real(complex_amplitudes[i] * Y_21[i])
                        ambisonic_ir[8, sample_idx] += np.real(complex_amplitudes[i] * Y_22[i])
#                    print('R channel (Order 2)', np.count_nonzero(ambisonic_ir[4]))
#                    print('S channel (Order 2)', np.count_nonzero(ambisonic_ir[5]))
#                    print('T channel (Order 2)', np.count_nonzero(ambisonic_ir[6]))
#                    print('U channel (Order 2)', np.count_nonzero(ambisonic_ir[7]))
#                    print('V channel (Order 2)', np.count_nonzero(ambisonic_ir[8]))

                # Apply windowing to smooth the IR (optional)
                window = np.hanning(ir_length)
                for ch in range(n_channels):
                    ambisonic_ir[ch] *= window

                # Normalize to prevent clipping
                max_val = np.max(np.abs(ambisonic_ir))
                if max_val > 0:
                    ambisonic_ir /= max_val

                # Transpose to get (n_samples, n_channels)
                ambisonic_ir_T = ambisonic_ir.T

                # Save the ambisonic_ir_T to multitrack WAV file
                output_dir = 'impulse_renspose'
                os.makedirs(output_dir, exist_ok=True)
                subtype='FLOAT'
                filename=f"{output_dir}/aIR_{src_idx}_{out_idx}_{bands_idx}.wav"
                sf.write(filename, ambisonic_ir_T, sample_rate, subtype=subtype)

                print(f"Saved multitrack WAV file: {filename}")
                print(f"Shape: {ambisonic_ir_T.shape} (samples, channels)")
                print(f"Sample rate: {sample_rate} Hz")
                print(f"Duration: {ambisonic_ir.shape[0] / sample_rate:.2f} seconds")
                print(f"Format: {subtype}")

    return ambisonic_ir

def convolve_mono_to_ambisonics(mono_audio, ambisonics_ir, method='fft'):
    """
    Convolve a mono audio track with an ambisonics impulse response.
    
    Parameters:
    -----------
    mono_audio : numpy.ndarray
        Mono audio track of shape (n_samples,) with dtype np.float32
    ambisonics_ir : numpy.ndarray
        Ambisonics impulse response of shape (n_channels, n_ir_samples) with dtype np.float32
    method : str, optional
        Convolution method: 'fft' (faster for long signals) or 'direct' (default: 'fft')
    
    Returns:
    --------
    ambisonics_output : numpy.ndarray
        Convolved ambisonics audio of shape (n_channels, n_output_samples) with dtype np.float32
    """
    
    # Input validation
    if mono_audio.ndim != 1:
        raise ValueError(f"mono_audio must be 1D array, got shape {mono_audio.shape}")
    
    if ambisonics_ir.ndim != 2:
        raise ValueError(f"ambisonics_ir must be 2D array, got shape {ambisonics_ir.shape}")
    
    n_channels, n_ir_samples = ambisonics_ir.shape
    n_audio_samples = len(mono_audio)
    
    # Check that we have first and second order (should be 4 for 1st order, 9 for 2nd order)
    expected_channels = {4: "1st order (4 channels: W, X, Y, Z)",
                        9: "2nd order (9 channels: W, X, Y, Z, R, S, T, U, V)"}
    
    if n_channels not in expected_channels:
        warnings.warn(f"Unexpected number of channels: {n_channels}. "
                     f"Expected {expected_channels}")
    
    # Initialize output array
    output_length = n_audio_samples + n_ir_samples - 1
    ambisonics_output = np.zeros((n_channels, output_length), dtype=np.float32)
    
    # Perform convolution convolution for each channel
    if method == 'fft':
        # Use FFT-based convolution (faster for long signals)
        for ch in range(n_channels):
            ambisonics_output[ch] = fftconvolve(mono_audio, ambisonics_ir[ch], mode='full')
    elif method == 'direct':
        # Use direct convolution (more accurate for short signals)
        for ch in range(n_channels):
            ambisonics_output[ch] = convolve(mono_audio, ambisonics_ir[ch], mode='full')
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'fft' or 'direct'")
    
    return ambisonics_output

def normalize_ambisonics_output(ambisonics_output, normalize_individually=False):
    """
    Optionally normalize the ambisonics output to prevent clipping.
    
    Parameters:
    -----------
    ambisonics_output : numpy.ndarray
        Ambisonics audio of shape (n_channels, n_samples)
    normalize_individually : bool, optional
        If True, normalize each channel independently. 
        If False, normalize based on the maximum across all channels (default: False)
    
    Returns:
    --------
    normalized_output : numpy.ndarray
        Normalized ambisonics audio
    """
    if normalize_individually:
        # Normalize each channel independently
        max_vals = np.max(np.abs(ambisonics_output), axis=1, keepdims=True)
        # Avoid division by zero
        max_vals[max_vals == 0] = 1.0
        normalized = ambisonics_output / max_vals
    else:
        # Normalize based on global maximum
        max_val = np.max(np.abs(ambisonics_output))
        if max_val > 0:
            normalized = ambisonics_output / max_val
        else:
            normalized = ambisonics_output
    
    return normalized.astype(np.float32)

def _compute_reflection_directions(incident_directions, normals, reflection_angles):
    """
    Compute reflection direction vectors from reflection angles.
    Parameters:
    - incident_directions: array of shape (n, 3) - incoming ray directions
    - normals: array of shape (n, 3) - surface normals
    - reflection_angles: array of shape (n,) - modified reflection angles
    Returns:
    - reflection_directions: array of shape (n, 3) - outgoing ray directions
    """
    # Normalize input vectors
    incident_directions = incident_directions / np.linalg.norm(incident_directions, axis=1, keepdims=True)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    # The reflection direction lies in the plane formed by the incident direction and normal
    # We need to rotate the incident direction around the normal by the reflection angle
    
    # First, compute the projection of the incident direction onto the normal
    n_dot_i = np.sum(normals * incident_directions, axis=1, keepdims=True)
    
    # Component of incident direction along the normal
    incident_normals = n_dot_i * normals
    
    # Component of incident direction perpendicular to the normal
    incident_tangent = incident_directions - incident_normals
    
    # Normalize the tangent component
    tangent_norm = np.linalg.norm(incident_tangent, axis=1, keepdims=True)
    incident_tangent_unit = incident_tangent / (tangent_norm + 1e-10)  # avoid division by zero
    
    # For a perfect reflection, the reflection direction would be:
    # reflection = incident_d_direction - 2 * n_dot_i * normals
    # But with scattering, we modify the angle. The reflection direction is:
    # reflection = cos(reflection_angles) * incident_normals - sin(reflection_angles) * incident_tangent_unit
    
    # Compute the reflection direction
    reflection_directions = (np.cos(reflection_angles) * incident_normals - np.sin(reflection_angles) * incident_tangent_unit)
    
    # Normalize the result
    reflection_directions = reflection_directions / np.linalg.norm(reflection_directions, axis=1, keepdims=True)
    return reflection_directions


def _random_hemisphere_directions(origins: np.ndarray, origins_idx: np.ndarray, origins_bands: np.ndarray, destinations: np.ndarray, destinations_idx: np.ndarray, scattered_energies: np.ndarray, scattered_phases: np.ndarray, delay: np.ndarray, normals: np.ndarray, roughness_factor: np.ndarray, max_scattering: int) -> np.ndarray:
    """
    Generate random directions on hemispheres oriented along the normals.
    Args:
        normals: Surface normals
        n_samples: Number of samples to generate
    Returns:
        Array of sampled directions
    """
    # Generate roughness dependent scattering rays number between 1 and max_scattering
    min_rayx = 1/roughness_factor
    max_rayx = max_scattering/roughness_factor
    n_scat_origins = origins.shape[0]
    n_scat_rays = np.random.randint(min_rayx, max_rayx, size=(n_scat_origins,1)) * roughness_factor
    n_scat_rays = n_scat_rays.astype(np.int32)
    n_samples = np.sum(n_scat_rays)

    # Generate scattering rays info array
    hi_idx = 0 
    scat_origins = np.zeros((n_samples,3), dtype=np.float32)
    scat_origins_idx = np.zeros((n_samples,1), dtype=np.int32)
    scat_origins_bands = np.zeros((n_samples,1), dtype=np.int32)
    scat_destinations = np.zeros((n_samples,3), dtype=np.float32)
    scat_destinations_idx = np.zeros((n_samples,1), dtype=np.int32)
    scat_normals = np.zeros((n_samples,3), dtype=np.float32)
    scat_energies = np.zeros((n_samples,1), dtype=np.float32)
    scat_phases = np.zeros((n_samples,1), dtype=np.float32)
    scat_delay = np.zeros((n_samples,1), dtype=np.float32)

    for idx in range(n_scat_rays.shape[0]):
        lo_idx = hi_idx if idx > 0 else 0
        hi_idx = lo_idx + int(n_scat_rays[idx])
        scat_origins[lo_idx:hi_idx] = origins[idx]
        scat_origins_idx[lo_idx:hi_idx] = origins_idx[idx]
        scat_origins_bands[lo_idx:hi_idx] = origins_bands[idx]
        scat_destinations[lo_idx:hi_idx] = destinations[idx]
        scat_destinations_idx[lo_idx:hi_idx] = destinations_idx[idx]
        scat_normals[lo_idx:hi_idx] = normals[idx]
        scat_energies[lo_idx:hi_idx] = scattered_energies[idx] / (hi_idx - lo_idx)
        scat_phases[lo_idx:hi_idx] = scattered_phases[idx]
        scat_delay[lo_idx:hi_idx] = delay[idx]

    scat_directions = np.random.uniform(-1,1,(n_samples,3))
    scat_directions /= np.linalg.norm(scat_directions)
    while not np.all(scat_directions[:, 0]**2 + scat_directions[:, 1]**2 + scat_directions[:, 2]**2 < 1):
        scat_directions = np.random.uniform(-1,1,(n_samples,3))
        # Project onto hemisphere oriented along normals
        scat_directions /= np.linalg.norm(directions)
    # Flip if pointing away from normals
    if not np.any(np.sum(scat_directions * scat_normals, axis=1) < 0):
        mask = np.sum(scat_directions * scat_normals, axis=1) < 0
        scat_directions[mask] = -scat_directions[mask]

    return scat_origins, scat_origins_idx, scat_origins_bands, scat_destinations, scat_destinations_idx, scat_energies, scat_phases, scat_delay, scat_directions

def _compute_acoustic_object_coefficients(c: float, rho: float, E: float, nu: float, damping: float, freq_bands: List[Tuple[float,float]]):
    """
    Calculate medium attenuation coefficient and phase shift for acoustic objects.
    Works for gases, fluids, and solids using a simplified common method.
    """
    # Compute Rayleigh damping coefficient α (mass proportional) and Rayleigh damping coefficient β (stiffness proportional)
    print('Compute Rayleigh damping coefficient')
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
destinations = np.zeros((0,3), dtype=np.float32)
destinations_idx = np.zeros((0,1), dtype=np.float32)
origins = np.zeros((0,3), dtype=np.float32)
origins_idx = np.zeros((0,1), dtype=np.int32)
origins_bands = np.zeros((0,1), dtype=np.int32)
output_energies = np.zeros((0,1), dtype=np.float32)
output_phases = np.zeros((0,1), dtype=np.float32)
output_delay = np.zeros((0,1), dtype=np.float32)
output_bands = np.zeros((0,1), dtype=np.float32)
output_source = np.zeros((0,1), dtype=np.float32)
output_origins = np.zeros((0,3), dtype=np.float32)
output_directions = np.zeros((0,3), dtype=np.float32)
output_destinations = np.zeros((0,3), dtype=np.float32)
output_destinations_idx = np.zeros((0,1), dtype=np.float32)
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

# Fill objects shader properties info with null values
roughness = 0
roughness_info = np.append(roughness_info, np.full((vertices[faces].shape[0],1), [roughness], dtype=np.float32), axis=0)
# Get material properties absorption
abs_coeffs = np.full((1,n_bands), [1], dtype=np.float32)
abs_phases = np.full((1,n_bands), [0], dtype=np.float32)
abs_coeffs_info = np.append(abs_coeffs_info, np.full((vertices[faces].shape[0],n_bands), abs_coeffs, dtype=np.float32), axis=0)
abs_phases_info = np.append(abs_phases_info, np.full((vertices[faces].shape[0],n_bands), abs_phases, dtype=np.float32), axis=0)
# Get material properties reflecrtion
refl_coeffs = np.full((1,n_bands), [0], dtype=np.float32)
refl_phases = np.full((1,n_bands), [0], dtype=np.float32)
refl_coeffs_info = np.append(refl_coeffs_info, np.full((vertices[faces].shape[0],n_bands), refl_coeffs, dtype=np.float32), axis=0)
refl_phases_info = np.append(refl_phases_info, np.full((vertices[faces].shape[0],n_bands), refl_phases, dtype=np.float32), axis=0)
# Get material properties refraction
refr_coeffs = np.full((1,n_bands), [0], dtype=np.float32)
refr_phases = np.full((1,n_bands), [0], dtype=np.float32)
refr_coeffs_info = np.append(refr_coeffs_info, np.full((vertices[faces].shape[0],n_bands), refr_coeffs, dtype=np.float32), axis=0)
refr_phases_info = np.append(refr_phases_info, np.full((vertices[faces].shape[0],n_bands), refr_phases, dtype=np.float32), axis=0)
# Get material properties refraction
scat_coeffs = np.full((1,n_bands), [0], dtype=np.float32)
scat_phases = np.full((1,n_bands), [0], dtype=np.float32)
scat_coeffs_info = np.append(scat_coeffs_info, np.full((vertices[faces].shape[0],n_bands), scat_coeffs, dtype=np.float32), axis=0)
scat_phases_info = np.append(scat_phases_info, np.full((vertices[faces].shape[0],n_bands), scat_phases, dtype=np.float32), axis=0)

n_rays = config.system.number_of_rays
num_points = n_rays * n_bands

# Output data and mesh
out_idx = -2
for out_config in config.outputs:
    out_idx += -1
    pose = np.load('data/pose/Camera.npz')
    output_pos = pose[pose.files[0]].reshape(-1,3)
    output_arr = np.full((num_points,3), [output_pos], dtype=np.float32)
    output_idx = np.full((num_points,1), [out_config.idx], dtype=np.float32)
    destinations = np.append(destinations, output_arr, axis=0)
    destinations_idx = np.append(destinations_idx, output_idx, axis=0)
    if out_config.size == 0:
        out_config.size = 0.1
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=out_config.size)
    mesh.apply_transform([[1, 0, 0, output_pos[0][0]],[0, 1, 0, output_pos[0][1]],[0, 0, 1, output_pos[0][2]],[0, 0, 0, 1]])
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)
    mesh_info = np.append(mesh_info, mesh.vertices[mesh.faces], axis=0)
    scene_info = np.append(scene_info, np.full((mesh.vertices[mesh.faces].shape[0],), [out_idx], dtype=np.int32))

    # Fill objects shader properties info with null values
    roughness = 0
    roughness_info = np.append(roughness_info, np.full((vertices[faces].shape[0],1), [roughness], dtype=np.float32), axis=0) 
    # Get material properties absorption
    abs_coeffs = np.full((1,n_bands), [1], dtype=np.float32)
    abs_phases = np.full((1,n_bands), [0], dtype=np.float32)
    abs_coeffs_info = np.append(abs_coeffs_info, np.full((vertices[faces].shape[0],n_bands), abs_coeffs, dtype=np.float32), axis=0) 
    abs_phases_info = np.append(abs_phases_info, np.full((vertices[faces].shape[0],n_bands), abs_phases, dtype=np.float32), axis=0) 
    # Get material properties reflecrtion
    refl_coeffs = np.full((1,n_bands), [0], dtype=np.float32)
    refl_phases = np.full((1,n_bands), [0], dtype=np.float32)
    refl_coeffs_info = np.append(refl_coeffs_info, np.full((vertices[faces].shape[0],n_bands), refl_coeffs, dtype=np.float32), axis=0) 
    refl_phases_info = np.append(refl_phases_info, np.full((vertices[faces].shape[0],n_bands), refl_phases, dtype=np.float32), axis=0) 
    # Get material properties refraction
    refr_coeffs = np.full((1,n_bands), [0], dtype=np.float32)
    refr_phases = np.full((1,n_bands), [0], dtype=np.float32)
    refr_coeffs_info = np.append(refr_coeffs_info, np.full((vertices[faces].shape[0],n_bands), refr_coeffs, dtype=np.float32), axis=0) 
    refr_phases_info = np.append(refr_phases_info, np.full((vertices[faces].shape[0],n_bands), refr_phases, dtype=np.float32), axis=0) 
    # Get material properties refraction
    scat_coeffs = np.full((1,n_bands), [0], dtype=np.float32)
    scat_phases = np.full((1,n_bands), [0], dtype=np.float32)
    scat_coeffs_info = np.append(scat_coeffs_info, np.full((vertices[faces].shape[0],n_bands), scat_coeffs, dtype=np.float32), axis=0) 
    scat_phases_info = np.append(scat_phases_info, np.full((vertices[faces].shape[0],n_bands), scat_phases, dtype=np.float32), axis=0)

# Source data
n_src = 0
source_bands = np.zeros((n_rays*n_bands,1), dtype=np.int32)
for src_config in config.sources:
    pose = np.load(f"{src_config.pose_path}/{src_config.name}.npz")
    source_pos = pose[pose.files[0]].reshape(-1,3)
    n_src += 1
    for idx in range(n_rays):
        lo_idx = n_bands * idx
        hi_idx = n_bands * (idx +1)
        source_bands[lo_idx:hi_idx] = np.arange(n_bands).reshape(-1,1)
    origins_bands = np.append(origins_bands, source_bands, axis=0)

    source_idx = np.full((num_points,1), [src_config.idx], dtype=np.int32)
    origins_idx = np.append(origins_idx, source_idx, axis=0)

    source_arr = np.full((num_points,3), [source_pos], dtype=np.float32)
    origins = np.append(origins, source_arr, axis=0)


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

main_dirs = destinations - origins
main_dirs_norm = np.linalg.norm(main_dirs, axis=1, keepdims=True)
main_dirs_norm[main_dirs_norm <= 1e-10] = 1e-10
main_dirs = main_dirs / main_dirs_norm

num_points = n_rays * n_src
num_dirs = n_rays * n_src * n_bands
directions = np.zeros((num_dirs,3), dtype=np.float32)

phi = np.pi * (3. - np.sqrt(5.))
theta = phi * np.arange(num_points)
z = np.linspace(1/num_points-1, 1-1/num_points, num_points)
radius = np.sqrt(1 - z * z)
y = radius * np.sin(theta)
x = radius * np.cos(theta)

dirs = np.array(list(zip(x,y,z)), dtype=np.float32)

for idx in range(n_rays):
    lo_idx = n_bands * idx
    hi_idx = n_bands * (idx +1)
    index = np.arange(lo_idx, hi_idx)
    directions[index] = dirs[idx]

for idx in range(n_src):
    main_idx = n_rays * n_bands * idx
    directions[main_idx] = main_dirs[main_idx]

energies = np.full((num_dirs,1), [1], dtype=np.float32)
phases = np.full((num_dirs,1), [0], dtype=np.float32)
delay = np.full((num_dirs,1), [0], dtype=np.float32)


loop(origins, origins_idx, origins_bands, destinations, destinations_idx, directions, energies, phases, delay, mesh_info, scene_info, recursion_idx, ad_alpha, ad_beta, medium_info_alpha, medium_info_beta, abs_coeffs_info, abs_phases_info, refl_coeffs_info, refl_phases_info, refr_coeffs_info, refr_phases_info, scat_coeffs_info, scat_phases_info, roughness_info, frequency_bands.get_bands(), output_source, output_bands, output_energies, output_phases, output_delay, output_origins, output_directions, output_destinations, output_destinations_idx)

