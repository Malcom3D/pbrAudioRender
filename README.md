## License

This project is licensed under the **GNU General Public License v3.0 or later**.

See the [LICENSE](LICENSE) file for the full text.

## pbrAudioRender

Physically based rendered Audio: 3D acoustic render.



          ^ z
          |
          |
          .-----------
         /|          /|
        / |   p     / |    ---> y
       /  | (center)/  |
      /   |        /   |
     .----|-------.    |
     |    |       |    |
     |    .-------|----.
     |   /        |   /
     |  / v_y     |  /
     | /          | /
     |/           |/
     .------------.
    /
   /x


Soxel: acoustic pressure, acoustic particle velocity and acoustic material properties in a voxel


References [not confirmed]:
- https://graphics.stanford.edu/projects/wavesolver/assets/wavesolver2018_opt.pdf as white paper resources.
- https://github.com/videolabs/libspatialaudio as ambisonic render bridge.
- https://www.acoular.org to generate 3D mappings of sound source from multichannel data recorded by a microphone array (ambisonic).


