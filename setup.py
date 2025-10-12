# Copyright (C) 2025 Malcom3D <malcom3d.gpl@gmail.com>
#
# This file is part of pbrAudio.
#
# pbrAudio is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pbrAudio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pbrAudio.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0-or-later

from setuptools import setup, find_packages

setup(
    name="pbr-audio-render",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "numba>=0.55.0",
        "zarr>=2.11.0",
        "dask>=2022.0.0",
        "trimesh>=3.15.0",
        "soundfile>=0.10.0",
    ],
    extras_require={
        'gpu': ['pyopencl>=2022.0.0', 'cupy-cuda11x>=10.0.0'],
        'openvdb': ['openvdb>=10.0.0'],
    },
    python_requires='>=3.8',
)
