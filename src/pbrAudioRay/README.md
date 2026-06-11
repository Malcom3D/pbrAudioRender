pbrAudioRay
===========

**Physically-Based Ray-Traced Acoustic Rendering Engine**

pbrAudioRay is a 3D geometric acoustic rendering engine built on EmbreeX, designed for offline audio simulation in static and animated 3D scenes.
It models sound propagation through complex environments with physically plausible acoustic behavior, supporting moving sources, listeners, and objects.

## Features

### Core Capabilities
- **Static and Animated Meshes**: Support for both rigid and deformable objects
- **Geometric Ray Tracing**: Uses EmbreeX for high-performance ray-triangle intersection
- **Time-Varying Acoustics**: Supports dynamic scenes with moving sources, listeners, and objects
- **Multi-Band Processing**: Frequency-dependent acoustic simulation with configurable bands per octave
- **Ambisonic Output**: Generates spatial audio up to second ambisonic order
- **Parallel Computing**: SIMD acceleration with Numba and Dask for distributed processing

### Acoustic Phenomena
- **Reflection**: Specular reflection with frequency-dependent coefficients
- **Absorption**: Frequency-dependent energy absorption at surfaces
- **Scattering**: Diffuse scattering based on surface roughness
- **Transmission**: Sound transmission through objects with Snell's law refraction
- **Diffraction**: Edge diffraction modeling
- **Doppler Effect**: Frequency shift from moving sources/listeners
- **Resonance**: Helmholtz resonators, room modes, and structural resonance
- **Dispersion**: Frequency-dependent wave speed variations

### Material System
- **Physically-Based Acoustic Materials**: Frequency-dependent absorption, reflection, transmission, and scattering coefficients
- **Multi-Layer Materials**: Support for complex material stacks

## Quick Start

### Basic Configuration
Create a JSON configuration file file:

```json
{
  "system": {
    "sample_rate": 48000,
    "bit_depth": 32,
    "fps": 24,
    "number_of_rays": 16,
    "bands_per_octave": 24,
    "lowest_frequency": 5,
    "higher_frequency": 24000.0
  },
  "acoustic_domain": {
    "type": "world",
    "geometry": [
      [-10, -10, -10],
      [10, 10, 10]
    ],
    "acoustic_shader": {
      "sound_speed": 343.0,
      "density": 1.225,
      "temperature": 20.0
    }
  },
  "sources": [
    {
      "idx": 0,
      "name": "main_source",
      "type": "spherical",
      "static": false,
      "audio_file": "input.wav",
      "pose_path": "./poses/"
    }
  ],
  "outputs": [
    {
      "idx": 0,
      "name": "main_output",
      "type": "AMBI",
      "static": false,
      "order": 1,
      "pose_path": "./poses/"
    }
  ],
  "objects": [
    {
      "idx": 0,
      "name": "wall",
      "obj_path": "./meshes/wall.npz",
      "pose_path": "./poses/",
      "static": true,
      "acoustic_shader": {
        "sound_speed": 3400.0,
        "density": 2400.0,
        "roughness": 0.1,
        "acoustic_properties": {
          "absorption": {
            "frequencies": [125, 250, 500, 1000, 2000, 4000],
            "coefficients": [0.02, 0.02, 0.03, 0.04, 0.05, 0.05]
          },
          "reflection": {
            "frequencies": [125, 250, 500, 1000, 2000, 4000],
            "coefficients": [0.98, 0.98, 0.97, 0.96, 0.95, 0.95]
          }
        }
      }
    }
  ]
}
```

### Running the Engine

```python
from pbrAudioRay.core.acoustic_engine import AcousticEngine
from pbrAudioRay.core.entity_manager import EntityManager

# Initialize entity manager with configuration
entity_manager = EntityManager("config.json")

# Create and run the acoustic engine
engine = AcousticEngine(entity_manager)
engine.compute()
```

## Architecture

### Core Components

```
pbrAudioRay/
├── core/
│   ├── acoustic_engine.py      # Main engine orchestrator
│   └── entity_manager.py       # Singleton entity registry
├── engine/
│   ├── wave_propagator.py      # Wave propagation solver
│   ├── ray_tracer.py           # EmbreeX ray tracing
│   ├── interface.py            # Interface manager
│   └── interfaces/
│       ├── absorption.py       # Absorption processing
│       ├── reflection.py       # Reflection processing
│       ├── scattering.py       # Scattering processing
│       └── transmission.py     # Transmission processing
├── sources/
│   ├── spherical_source.py     # Omnidirectional source
│   └── planar_source.py        # Directional source
├── outputs/
│   ├── ambisonic_output.py     # Ambisonic microphone array
│   ├── omnidirectional_output.py # Omnidirectional microphone
│   ├── cardioid_output.py      # Cardioid microphone
│   ├── hypercardioid_output.py # HyperCardioid microphone
│   └── figure8_output.py       # Figure-8 microphone
├── lib/
│   ├── acoustic_shader.py      # Material property definitions
│   ├── acoustic_object.py      # 3D object handler
│   ├── ambisonic_convolver.py   # Time-varying convolution
│   ├── ambisonic_ir_interpolator.py # Ambisonic IR interpolator
│   ├── frequency_bands.py      # Frequency band generation
│   ├── frequency_response.py   # Spatial frequency response
│   ├── interpolator.py         # Frequency/spatial interpolation
│   ├── filter.py               # Linkwitz-Riley filters
│   ├── ray_data.py             # Ray tracing data structures
│   ├── output_data.py          # Output data accumulation store
│   ├── geometry_data.py        # Scene geometry
│   ├── material_properties.py  # Per-face material data
│   ├── medium_properties.py    # Propagation medium data
│   └── functions.py            # Utility functions
└── utils/
    └── config.py               # Configuration system
```

### Data Flow

1. **Configuration** → JSON file parsed into typed dataclasses
2. **Scene Initialization** → Geometry, materials, and medium properties loaded
3. **Ray Generation** → Sources emit rays with Fibonacci sphere distribution
4. **Ray Tracing** → EmbreeX accelerates ray-scene intersection
5. **Interface Processing** → Absorption, reflection, scattering, transmission at each hit
6. **Output Accumulation** → Energy, phase, and direction collected at listener positions
7. **Impulse Response** → Per-band IRs encoded to ambisonic format
8. **Audio Rendering** → Time-varying convolution with source audio

## Configuration Reference

### System Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 48000 | Audio sample rate (Hz) |
| `bit_depth` | 32 | Audio bit depth |
| `fps` | 24 | Video frame rate |
| `number_of_rays` | 16 | Rays per source per frame |
| `bands_per_octave` | 24 | Frequency resolution |
| `lowest_frequency` | 5 | Minimum simulation frequency (Hz) |
| `higher_frequency` | 24000 | Maximum simulation frequency (Hz) |

### Wave Propagation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_interactions` | 8192 | Maximum ray bounces |
| `enable_resonance` | true | Enable resonance effects |
| `use_dispersion_correction` | true | Frequency-dependent speed variations |
| `use_extended_reaction` | false | Extended reaction boundary conditions |

### Interface Properties
| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_reflection` | 5 | Maximum reflection bounces |
| `max_scattering` | 5 | Maximum scattering bounces |
| `max_transmission` | 0.75 | Maximum transmission fraction |
| `max_diffraction` | 5 | Maximum diffraction order |
| `enable_absorption` | true | Enable surface absorption |
| `enable_reflection` | true | Enable specular reflection |
| `enable_scattering` | true | Enable diffuse scattering |
| `enable_transmission` | true | Enable sound transmission |

## Material Properties

### Acoustic Shader Parameters
| Parameter | Units | Description |
|-----------|-------|-------------|
| `sound_speed` | m/s | Speed of sound in material |
| `density` | kg/m³ | Material density |
| `young_modulus` | Pa | Young's modulus (structural) |
| `poisson_ratio` | - | Poisson's ratio (structural) |
| `damping` | - | Rayleigh damping coefficient |
| `roughness` | - | Surface roughness (0-1) |
| `impedance` | Rayl | Characteristic impedance |
| `temperature` | °C | Material temperature |

### Frequency-Dependent Coefficients
Each acoustic property supports frequency-dependent values:

```json
{
  "acoustic_properties": {
    "absorption": {
      "frequencies": [125, 250, 500, 1000, 2000, 4000],
      "coefficients": [0.1, 0.2, 0.5, 0.8, 0.9, 0.95],
      "phases": [0, 0, 0, 0, 0, 0]
    },
    "reflection": {
      "frequencies": [125, 250, 500, 1000, 2000, 4000],
      "coefficients": [0.9, 0.8, 0.5, 0.2, 0.1, 0.05]
    },
    "transmission": {
      "frequencies": [125, 250, 500, 1000, 2000, 4000],
      "coefficients": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    },
    "scattering": {
      "frequencies": [125, 250, 500, 1000, 2000, 4000],
      "coefficients": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    }
  }
}
```

## Performance Optimization

### Ray Count Tuning
- **Low quality**: 4-8 rays per source
- **Medium quality**: 16-32 rays per source
- **High quality**: 64-128 rays per source
- **Production**: 256+ rays per source

### Frequency Resolution
- **Low**: 0 bands per octave (fast)
- **Medium**: 1 bands per octave (balanced)
- **High**: 3 bands per octave (detailed)
- **Ultra**: 16 bands per octave (research)

### Parallel Processing
```python
# Configure Dask workers
from dask import config as dask_config
dask_config.set(num_workers=1024)

# Configure Numba threads
import numba as nb
nb.set_num_threads(8)
```

## Output Formats

### Ambisonic
- Supports up to 2nd ambisonic order
- ACN channel ordering
- SN3D normalization
- Output formats: WAV, BWF, RAW

### Mono Outputs
- Omnidirectional
- Cardioid
- Hypercardioid
- Figure-8

### File Formats
- `.wav`: Standard WAV with PCM or float
- `.bwf`: Broadcast Wave Format with metadata
- `.raw`: Raw float32 data
- `.npz`: NumPy compressed archive

### Moving Source with Doppler
```python
# Source poses should contain position and rotation per frame
# Format: npz file with 'positions' and 'rotations' arrays
pose_data = {
    'positions': np.array([
        [0, 0, 0],    # Frame 0
        [1, 0, 0],    # Frame 1
        [2, 0, 0],    # Frame 2
        # ... per frame positions
    ]),
    'rotations': np.array([
        [0, 0, 0],    # Frame 0 (yaw, pitch, roll)
        [0, 0, 0],    # Frame 1
        [0, 0, 0],    # Frame 2
        # ... per frame rotations
    ])
}
np.savez('poses/source.npz', **pose_data)
```

## License

This project is licensed under the **GNU General Public License v3.0 or later**.

See the [LICENSE](LICENSE) file for the full text.
