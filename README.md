# PbrAudioRender

### Physically Based Rendering (PBR) for Audio

**PbrAudioRender** is a sophisticated audio rendering engine that implements a physically based rendering (PBR) model to simulate the interaction of sound waves with environments. The engine processes audio signals using a Finite-Difference Time-Domain (FDTD) model, simulating how sound waves travel, reflect, refract, diffuse, scatter, and are absorbed by different materials with realistic physical accuracy.

## Overview

The Acoustic Rendering Engine takes audio signals and geometric data as input, producing realistic audio output by simulating how sound travels and interacts with materials. The PBR approach models the physical properties of materials that affect sound behavior.

## Key Features

### Sound Propagation Simulation
- **Distance Attenuation:** Models how sound intensity decreases with distance
- **Reflections:** Simulates sound bouncing off surfaces
- **Diffraction:** Models sound bending around corners and obstacles
- **Refraction:** Simulates sound bending as it passes through different materials

### Material Properties Modeling
- **Absorption Coefficients:** Controls how much sound energy is absorbed by materials
- **Reflection Coefficients:** Determines how much sound energy is reflected
- **Roughness/Smoothness:** Affects sound scattering behavior (rough surfaces scatter sound diffusely, smooth surfaces reflect cleanly)
- **Density:** Influences sound wave propagation characteristics
- **Speed of Sound:** Models material-specific sound velocity variations

## Input Data

- **Raw Audio Samples:** Processes audio data from WAV files
- **Geometric Data:** Requires scene geometry information including:
  - Sound source and listener positions
  - Object positions and orientations
  - Material property assignments
  - Environmental boundaries

## Output Data

The engine produces rendered audio signals that simulate realistic acoustic environments, suitable for playback through speakers or headphones.

## Applications

- **Games and VR/AR:** Creating immersive audio environments for interactive experiences
- **Audio Post-Production:** Simulating acoustic spaces spaces for film, music, and multimedia projects
- **Architectural Acoustics:** Predicting sound behavior in buildings and rooms
- **Scientific Research:** Studying sound propagation and material interactions

## References

- [Wave Solver for Sound Propagation](https://graphics.stanford.edu/projects/wavesolver/assets/wavesolver2018_opt.pdf) - White paper resources

## License

This project is licensed under the **GNU General Public License v3.0 or later**.

See the [LICENSE](LICENSE) file for the full text.

