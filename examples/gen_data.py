#!/usr/bin/env python3
"""
Data generation script for pbrAudioRender acoustic simulation system.
Generates all necessary input data including audio files, position data, 
directivity patterns, and 3D objects.
"""

import os
import numpy as np
import soundfile as sf
from scipy import signal
import json
from typing import List, Tuple, Dict, Any

def create_directories():
    """Create all necessary directories for the simulation"""
    directories = [
        './data/audio/',
        './data/positions/',
        './data/directivity/',
        './data/calibration/',
        './data/objects/',
        './pbrAudioCache/filtered_audio/',
        './exports/audio/',
        './exports/ambisonic/',
        './exports/vdb/'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def generate_test_audio(duration: float = 2.0, sample_rate: int = 48000):
    """Generate test audio signals for sources"""
    print("Generating test audio files...")
    
    # Source 1: Sine sweep
    t = np.linspace(0, duration, int(sample_rate * duration))
    freq_sweep = 20 + 19000 * (t / duration)  # 100Hz to 20kHz sweep
    source1_audio = 0.5 * np.sin(2 * np.pi * freq_sweep * t)
    
    # Apply envelope to avoid clicks
    envelope = np.ones_like(t)
    attack_samples = int(0.01 * sample_rate)  # 10ms attack
    release_samples = int(0.1 * sample_rate)  # 100ms release
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    source1_audio *= envelope
    
    # Source 2: Pink noise burst
    source2_audio = generate_pink_noise(len(t))
    # Apply window to noise
    source2_audio *= np.hanning(len(t))
    
    # Save audio files
    sf.write('./data/audio/source_1.wav', source1_audio, sample_rate)
    sf.write('./data/audio/source_2.wav', source2_audio, sample_rate)
    
    print("Generated audio files: source_1.wav, source_2.wav")

def generate_pink_noise(length: int) -> np.ndarray:
    """Generate pink noise using Voss-McCartney algorithm"""
    num_generators = 12
    generators = [np.random.randn(length) for _ in range(num_generators)]
    
    pink_noise = np.zeros(length)
    for i in range(length):
        for j in range(num_generators):
            if i % (2 ** j) == 0:
                pink_noise[i] = generators[j][i]
    
    # Normalize
    pink_noise = pink_noise / np.max(np.abs(pink_noise)) * 0.5
    return pink_noise

def generate_position_data(num_frames: int = 1000):
    """Generate position data for sources and listener"""
    print("Generating position data...")
    
    # Source 1: Moving in a circle
    t = np.linspace(0, 4 * np.pi, num_frames)
    radius = 1.0
    source1_x = 2.0 + radius * np.cos(t)
    source1_y = 4.0 + radius * np.sin(t)
    source1_z = 1.0 + 0.2 * np.sin(2 * t)
    source1_positions = np.column_stack([source1_x, source1_y, source1_z])
    
    # Source 2: Static position
    source2_positions = np.tile([8.0, 4.0, 1.0], (num_frames, 1))
    
    # Listener: Slight head movement
    listener_x = 5.0 + 0.1 * np.sin(0.5 * t)
    listener_y = 4.0 + 0.05 * np.cos(0.5 * t)
    listener_z = 1.5 * t
    listener_positions = np.column_stack([listener_x, listener_y, listener_z])
    
    # Ambisonic array positions (4 channels in tetrahedron)
    ambisonic_positions = []
    for i in range(4):
        angle = i * np.pi / 2
        x = 5.0 + 0.1 * np.cos(angle)
        y = 4.0 + 0.1 * np.sin(angle)
        z = 1.5
        channel_positions = np.column_stack([
            x + 0.01 * np.sin(0.3 * t),
            y + 0.01 * np.cos(0.3 * t),
            np.full(num_frames, z)
        ])
        ambisonic_positions.append(channel_positions)
    
    # Save position data
    np.savez_compressed('./data/positions/source_1.npz', source1_positions)
    np.savez_compressed('./data/positions/source_2.npz', source2_positions)
    np.savez_compressed('./data/positions/listener.npz', listener_positions)
    np.savez_compressed('./data/positions/ambisonic_array.npz', np.array(ambisonic_positions))
    
    print("Generated position files for sources, listener, and ambisonic array")

def generate_directivity_patterns():
    """Generate directivity patterns for speakers and microphones"""
    print("Generating directivity patterns...")
    
    # Speaker directivity (cardioid pattern that becomes more directional at high frequencies)
    azimuths = np.linspace(0, 360, 37)  # 10 degree steps
    elevations = np.linspace(-90, 90, 19)  # 10 degree steps
    frequencies = np.array([20, 100, 500, 1000, 5000, 20000])
    
    speaker_responses = []
    speaker_phases = []
    
    for az_idx, az in enumerate(azimuths):
        elev_responses = []
        elev_phases = []
        for el_idx, el in enumerate(elevations):
            freq_responses = []
            freq_phases = []
            for freq_idx, freq in enumerate(frequencies):
                # Cardioid pattern: 0.5 * (1 + cos(theta))
                # More directional at high frequencies
                az_rad = np.deg2rad(az)
                el_rad = np.deg2rad(el)
                
                # Base cardioid response
                cardioid = 0.5 * (1 + np.cos(az_rad) * np.cos(el_rad))
                
                # Frequency-dependent directivity
                freq_factor = 1.0 + 0.5 * (freq / 10000)  # More directional at high frequencies
                response = cardioid ** (1.0 / freq_factor)
                
                # Simple phase response (minimal phase shift)
                phase = 0.1 * np.sin(az_rad) * (freq / 1000)
                
                freq_responses.append(response)
                freq_phases.append(phase)
            
            elev_responses.append(freq_responses)
            elev_phases.append(freq_phases)
        
        speaker_responses.append(elev_responses)
        speaker_phases.append(elev_phases)
    
    # Omnidirectional microphone response
    omni_responses = []
    omni_phases = []
    
    for az_idx, az in enumerate(azimuths):
        elev_responses = []
        elev_phases = []
        for el_idx, el in enumerate(elevations):
            freq_responses = []
            freq_phases = []
            for freq_idx, freq in enumerate(frequencies):
                # Omnidirectional: constant response
                response = 1.0
                # Minimal phase variation
                phase = 0.01 * (freq / 1000)
                
                freq_responses.append(response)
                freq_phases.append(phase)
            
            elev_responses.append(freq_responses)
            elev_phases.append(freq_phases)
        
        omni_responses.append(elev_responses)
        omni_phases.append(elev_phases)
    
    # Save directivity patterns
    speaker_data = {
        'azimuths': azimuths.tolist(),
        'elevations': elevations.tolist(),
        'frequencies': frequencies.tolist(),
        'responses': speaker_responses,
        'phases': speaker_phases
    }
    
    omni_data = {
        'azimuths': azimuths.tolist(),
        'elevations': elevations.tolist(),
        'frequencies': frequencies.tolist(),
        'responses': omni_responses,
        'phases': omni_phases
    }
    
    np.savez_compressed('./data/directivity/speaker_directivity.npz', speaker_data)
    np.savez_compressed('./data/directivity/omni_mic.npz', omni_data)
    
    print("Generated directivity patterns for speaker and omnidirectional microphone")

def generate_calibration_data():
    """Generate microphone calibration data"""
    print("Generating calibration data...")
    
    azimuths = np.linspace(0, 360, 37)
    elevations = np.linspace(-90, 90, 19)
    frequencies = np.array([20, 100, 500, 1000, 5000, 20000])
    
    calibration_responses = []
    calibration_phases = []
    
    # Flat frequency response with slight high-frequency rolloff
    for az_idx, az in enumerate(azimuths):
        elev_responses = []
        elev_phases = []
        for el_idx, el in enumerate(elevations):
            freq_responses = []
            freq_phases = []
            for freq_idx, freq in enumerate(frequencies):
                # Slight high-frequency rolloff
                response = 1.0 / (1.0 + (freq / 15000) ** 2)
                # Minimal phase correction
                phase = -0.05 * (freq / 1000)
                
                freq_responses.append(response)
                freq_phases.append(phase)
            
            elev_responses.append(freq_responses)
            elev_phases.append(freq_phases)
        
        calibration_responses.append(elev_responses)
        calibration_phases.append(elev_phases)
    
    calibration_data = {
        'azimuths': azimuths.tolist(),
        'elevations': elevations.tolist(),
        'frequencies': frequencies.tolist(),
        'responses': calibration_responses,
        'phases': calibration_phases
    }
    
    np.savez_compressed('./data/calibration/mic_calibration.npz', calibration_data)
    print("Generated microphone calibration data")

def generate_simple_objects():
    """Generate simple 3D object files for testing"""
    print("Generating 3D object files...")
    
    # Simple table object
    table_vertices = [
        [1.0, 2.0, 0.0], [3.0, 2.0, 0.0], [3.0, 6.0, 0.0], [1.0, 6.0, 0.0],  # Bottom
        [1.0, 2.0, 0.8], [3.0, 2.0, 0.8], [3.0, 6.0, 0.8], [1.0, 6.0, 0.8],  # Top
        [0.8, 1.8, 0.0], [0.8, 1.8, 0.8], [3.2, 1.8, 0.0], [3.2, 1.8, 0.8],  # Legs
        [0.8, 6.2, 0.0], [0.8, 6.2, 0.8], [3.2, 6.2, 0.0], [3.2, 6.2, 0.8]   # Legs
    ]
    
    table_faces = [
        [1, 2, 3], [1, 3, 4],  # Bottom
        [5, 7, 6], [5, 8, 7],  # Top
        [1, 5, 2], [2, 5, 6],  # Sides
        [2, 6, 3], [3, 6, 7],
        [3, 7, 4], [4, 7, 8],
        [4, 8, 1], [1, 8, 5],
        # Legs
        [9, 10, 11], [11, 10, 12],
        [13, 14, 15], [15, 14, 16]
    ]
    
    # Room walls
    wall_vertices = [
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 8.0, 0.0], [0.0, 8.0, 0.0],  # Floor
        [0.0, 0.0, 6.0], [10.0, 0.0, 6.0], [10.0, 8.00, 6.0], [0.0, 8.0, 6.0],  # Ceiling
    ]
    
    wall_faces = [
        [1, 2, 3], [1, 3, 4],  # Floor
        [5, 7, 6], [5, 8, 7],  # Ceiling
        [1, 5, 2], [2, 5, 6],  # Walls
        [2, 6, 3], [3, 6, 7],
        [3, 7, 4], [4, 7, 8],
        [4, 8, 1], [1, 8, 5]
    ]
    
    # Save OBJ files
    with open('./data/objects/table.obj', 'w') as f:
        f.write("# Table object")
        for vertex in table_vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in table_faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    with open('./data/objects/walls.obj', 'w') as f:
        f.write("# Room walls object")
        for vertex in wall_vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in wall_faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print("Generated 3D object files: table.obj, walls.obj")

def validate_config():
    """Validate the configuration file"""
    print("Validating configuration...")
    
    try:
        with open('./config.json', 'r') as f:
            config = json.load(f)
        
        # Basic validation checks
        required_sections = ['system', 'gpu', 'acoustic_domain', 'sources', 'outputs']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Check acoustic domain geometry
        geometry = config['acoustic_domain']['geometry']
        if len(geometry) != 8:
            raise ValueError("Acoustic domain geometry must have 8 vertices")
        
        print("Configuration validation passed!")
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        raise

def main():
    """Main function to generate all data"""
    print("Starting data generation for pbrAudioRender...")
    
    # Create directory structure
    create_directories()
    
    # Generate all required data
    generate_test_audio()
    generate_position_data()
    generate_directivity_patterns()
    generate_calibration_data()
    generate_simple_objects()
    
    # Validate configuration
    validate_config()
    
    print("\n" + "="*50)
    print("Data generation completed successfully!")
    print("Generated files:")
    print("- Audio: ./data/audio/source_1.wav, source_2.wav")
    print("- Positions: ./data/positions/*.npz")
    print("- Directivity: ./data/directivity/*.npz")
    print("- Calibration: ./data/calibration/mic_calibration.npz")
    print("- 3D Objects: ./data/objects/table.obj, walls.obj")
    print("- Configuration: ./config.json")
    print("\nYou can now run the simulation with:")
    print("from pbrAudioRender import pbrAudioRender")
    print("simulation = pbrAudioRender('./('./config.json')")
    print("simulation.run_simulation()")
    print("="*50)

if __name__ == "__main__":
    main()

