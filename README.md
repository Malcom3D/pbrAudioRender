pbrAudioRender
===============

The `pbrAudioRender` is a Python framework comprising 3D acoustic engines designed for physically based sound synthesis and wave propagation rendering.
It is tailored for artists and sound designers seeking realistic and physically plausible acoustic simulations in complex environments.


## Features

Artist-Focused: Built with sound designers and creative professionals in mind.
Modular Architecture: Mix and match acoustic engines to meet your needs.
Physically Accurate: Based on real-world acoustic principles and physics.
Multi-Source Support: Handle sounds from various energy domains.
Scalable Performance: Ready to run from workstation to cloud (thanks to dask).


## Components

### pbrAudioShaders: Physically Based Synthesis
A collection of physically based acoustic shaders engine designed to synthesize realistic sound and noise effects based on physical principles from meshes interactions.
[Learn more](https://github.com/Malcom3D/pbrAudioShaders)

### pbrAudioRay: Geometric Acoustic Rendering
An acoustic geometric renderer engine built on EmbreeX ray tracing technology allowing accurate simulation of sound propagation through complex 3D enviroment geometries.
[Learn more](https://github.com/Malcom3D/pbrAudioRender/tree/main/src/pbrAudioRay)

### pbrAudioFDTD
An optimized Finite-Difference Time-Domain renderer engine that bridges near-field acoustic pressure generated from various energy phenomena (like thermal, aero, electrical, etc), with the the rest of the pbrAudioRender framework.
[Learn more](https://github.com/Malcom3D/pbrAudioRender/tree/main/src/pbrAudioFDTD)


## License

This project is licensed under the **GNU General Public License v3.0 or later**.

