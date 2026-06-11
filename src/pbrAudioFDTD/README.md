pbrAudioFDTD
============

## Optimized Finite-Difference Time-Domain Acoustic Render Engine

pbrAudioFDTD is a high-performance Finite-Difference Time-Domain (FDTD) acoustic simulation engine that bridges near-field sound generation from physical phenomena (thermal, mechanical, electrical, etc.) to the pbrAudioRender framework.
It provides physically accurate sound propagation, reflection, refraction, diffraction, scattering, and absorption simulation for realistic acoustic environments.

## Features

### Physical Sound Propagation
- **Multi-band FDTD Solver**: Frequency-dependent wave propagation with adaptive time-stepping
- **3D Spatial Acoustics**: Full 3D wave simulation with sub-voxel accuracy
- **Multi-layer Architecture**: Independent layer management for reflections and wave components
- **Courant Stability**: Automatic stability using audio clock as atomic unit

### Material Interactions
- **Absorption**: Frequency-dependent material absorption coefficients
- **Reflection**: Specular reflection with surface normal calculation
- **Refraction**: Snell's law implementation for sound speed discontinuities
- **Diffraction**: Uniform Theory of Diffraction (UTD) for edge diffraction
- **Scattering**: Diffuse scattering at rough surfaces
- **Impedance Matching**: Acoustic impedance ratio handling for material boundaries

### Acoustic Phenomena
- **Resonance Detection**: Helmholtz resonance, parallel wall resonance, tube resonance
- **Damping Control**: Adaptive energy damping to prevent numerical instability
- **Boundary Conditions**: Open (absorbing), periodic, and rigid boundary support
- **Energy Termination**: Automatic simulation termination based on energy thresholds

### Performance Optimizations
- **Numba JIT Compilation**: GPU-accelerated computation for critical kernels
- **Dask Parallel Processing**: Distributed computation for multi-band solvers
- **Adaptive Time-stepping**: Frequency-dependent time step optimization
- **Efficient Memory Management**: Sparse grid representation for acoustic properties

## Architecture

### Core Components

### Core Components

```
pbrAudioFDTD/
├── engine/
│     ├── wave_propagator.py      # Main propagation coordinator
│   ├── fdtd_solver.py          # Multi-band FDTD solver
│   ├── layer_manager.py        # Multi-layer field management
│   ├── damping.py              # Energy damping control
│   ├── boundary_conditions.py  # Domain boundary handling
│   ├── termination.py          # Simulation termination logic
│   ├── interface.py            # Interface interaction manager
│   ├── resonance.py            # Resonance phenomena manager
│   └── interfaces/
│       ├── absorption.py       # Material absorption
│       ├── reflection.py       # Specular reflection
│       ├── refraction.py       # Snell's law refraction
│       ├── diffraction.py      # UTD diffraction
│       └── scattering.py       # Diffuse scattering
├── sources/
│   ├── spherical_source.py     # Directional spherical sources
│   └── planar_source.py        # Directional planar sources
├── objects/
│   └── acoustic_object.py      # Voxelized acoustic objects
├── outputs/
│   ├── base_output.py          # Common microphone processing
│   ├── ambisonic_output.py     # Ambisonic microphone array
│   ├── omnidirectional_output.py # Omnidirectional microphone
│   ├── cardioid_output.py      # Cardioid microphone
│   ├── hypercardioid_output.py # HyperCardioid microphone
│   └── figure8_output.py       # Figure-8 microphone
├── lib/
│   ├── acoustic_shader.py      # Material property definitions
│   ├── acoustic_field.py       # Frequency-dependent field data
│   ├── acoustic_layer.py       # Layer field containers
│   ├── soxel.py               # Sound voxel definition
│   ├── filter.py              # Linkwitz-Riley filters
│   ├── interpolator.py        # 3D spatial interpolation
│   └── functions.py           # Utility functions
├── core/
│   ├── entity_manager.py       # Singleton/entity registry
│   ├── soxel_grid.py           # 3D voxel grid management
│   ├── acoustic_engine.py      # Simulation orchestration
│   └── audio_recorder.py       # Microphone recording
├── renderer/
│   ├── ambisonic_render.py     # Ambisonic B-format encoding
│   └── mono_render.py          # Mono audio rendering
└── utils/
    └── config.py               # Configuration management
```

### Data Flow

1. **Configuration**: JSON-based configuration defines acoustic domain, sources, objects, and outputs
2. **Grid Initialization**: SoxelGrid creates a 3D voxel grid with acoustic properties
3. **Source Injection**: Sources inject frequency-dependent pressure fields into the grid
4. **WaveWave Propagation**: FDTD solver propagates waves through the medium
5. **Interface Interactions**: Material boundaries trigger absorption, reflection, refraction, diffraction, and scattering
6. **Output Recording**: Microphones record pressure and velocity at specified positions
7. **Audio Rendering**: Recorded data is rendered to mono or ambisonic audio files

## Configuration

### Basic Setup

```json
{
  "system": {
    "frame_limit": 1000,
    "max_workers": 4,
    "cache_path": "./pbrAudioCache/"
  },
  "acoustic_domain": {
    "voxel_size": 0.1,
    "geometry": [[0,0,0], [10,0,0], [10,10,0], [0,10,0], [0,0,10], [10,0,10], [10,10,10], [0,10,10]],
    "sample_rate": 48000,
    "acoustic_shader": {
      "sound_speed": 343.0,
      "density": 1.225,
      "acoustic_properties": {
        "absorption": {
          "frequencies": [100, 1000, 10000],
          "coefficients": [0.1, 0.3, 0.5]
        }
      }
    }
  },
  "sources": [
    {
      "idx": 0,
      "name": "explosion",
      "type": "spherical",
      "geometry": 0.5,
      "audio_file": "explosion.wav",
      "position_file": "positions.npz",
      "acoustic_shader": {
        "sound_speed": 500.0,
        "density": 1.8,
        "acoustic_properties": {
          "absorption": {"frequencies": [100, 1000], "coefficients": [0.2, 0.4]},
          "scattering": {"frequencies": [100, 1000], "coefficients": [0.1, 0.3]}
        }
      }
    }
  ],
  "objects": [
    {
      "idx": 0,
      "name": "wall",
      "obj_path": "wall.obj",
      "acoustic_shader": {
        "sound_speed": 3000.0,
        "density": 2400.0,
        "acoustic_properties": {
          "reflection": {"frequencies": [100, 1000], "coefficients": [0.9, 0.7]},
          "absorption": {"frequencies": [100, 1000], "coefficients": [0.1, 0.3]}
        }
      }
    }
  ],
  "outputs": [
    {
      "idx": 0,
      "name": "listener",
      "type": "omnidirectional",
      "position_file": "listener_positions.npz"
    }
  ],
  "wave_propagation": {
    "enable_damping": true,
    "enable_boundary": true,
    "max_interactions": 5,
    "damping_coefficient": 0.02
  },
  "fdtd": {
    "courant_number": 0.5,
    "max_sound_speed": 500.0,
    "lowest_frequency": 5,
    "bands_per_octave": 24
  }
}
```

## Usage

### Basic Simulation

```python
from pbrAudioFDTD import pbrAudioRender

# Initialize with configuration
renderer = pbrAudioRender("config.json")

# Run simulation
renderer.start()
```

### Programmatic API

```python
from pbrAudioFDTD.core.entity_manager import EntityManager
from pbrAudioFDTD.core.acoustic_engine import AcousticEngine
from pbrAudioFDTD.core.audio_recorder import AudioRecorder

# Initialize
em = EntityManager("config.json")
engine = AcousticEngine(em)
recorder = = AudioRecorder(em)

# Custom simulation loop
frames = em.get('frames')
while frames.get() < frames.get_limit():
    engine.update()
    recorder.update()
    frames.next()
```

## Physical Models

### Wave Propagation
- **FDTD Method**: Second-order accurate finite differences in space and time
- **Multi-band Decomposition**: Frequency band splitting for frequency-dependent materials
- **Dispersion Correction**: Configurable dispersion compensation for numerical accuracy

### Material Properties
- **Acoustic Shader**: Defines sound speed, density, damping, and frequency-dependent coefficients
- **Acoustic Coefficients**: Interpolated absorption, reflection, refraction, and scattering coefficients
- **Impedance Matching**: Automatic handling of impedance discontinuities at material boundaries

### Source Models
- **Spherical Sources**: Directional spherical wave emission with configurable radius
- **Planar Sources**: Directional planar wave emission with arbitrary polygon geometry
- **Field Injection**: Frequency-dependent pressure and velocity field injection

### Output Processing
- **Microphone Models**: Omnidirectional, cardioid, hypercardioid, figure-8 patterns
- **Ambisonic Encoding**: N3D/SN3D normalized B-format encoding up to 3rd order
- **Sub-voxel Accuracy**: Trilinear interpolation for precise recording positions


## License

This project is licensed under the **GNU General Public License v3.0 or later**.

See the [LICENSE](LICENSE) file for the full text.
