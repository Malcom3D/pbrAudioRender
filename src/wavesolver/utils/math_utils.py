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

import numpy as np
import numba

@numba.jit(nopython=True)
def laplacian_3d(field: np.ndarray, dx: float) -> np.ndarray:
    """Compute 3D Laplacian using finite differences."""
    laplacian = np.zeros_like(field)
    nx, ny, nz = field.shape
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                d2f_dx2 = (field[i+1, j, k] - 2 * field[i, j, k] + field[i-1, j, k]) / (dx * dx)
                d2f_dy2 = (field[i, j+1, k] - 2 * field[i, j, k] + field[i, j-1, k]) / (dx * dx)
                d2f_dz2 = (field[i, j, k+1] - 2 * field[i, j, k] + field[i, j, k-1]) / (dx * dx)
                
                laplacian[i, j, k] = d2f_dx2 + d2f_dy2 + d2f_dz2
    
    return laplacian
