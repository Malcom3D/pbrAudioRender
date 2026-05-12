# Copyright (C) 2025 sheliak <janez.kos@gmail.com>
#
# This file is part of https://github.com/sheliak/varconvolve.
#
# VarConvolve is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# VarConvolve is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with VarConvolve.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.interpolate import interp1d
import types

def varconvolve(x,y,kernel,var,oversample=1,mode='same'):
    """
    x: an array with x coordinates of the points
    y: ana rray with y coordinates of the points
    kernel: name of the function that describes the kernel. Must have one argument, the width of the kernel in x units
    var: a function that returns the kernel width in one point.
    """
    if isinstance(var, (types.FunctionType)):
        pass
    elif isinstance(var, (list, tuple, np.ndarray)):
        var=interp1d(x,var)
    else:
        raise RuntimeError('var must be a function, a list, a tuple, or an ndarray')
    #check if sampling is uniform:
    if abs(np.max(np.diff(x))-np.min(np.diff(x)))<0.000001*np.diff(x)[0]:
        sampl=np.diff(x)[0]
    else:
        raise RuntimeError('Sampling must be uniform.')
    #build the new sampling array. Start at first element in x, end after we reach the last element in x:
    x_new=[x[0]]
    n=0
    m=np.max(var(x))
    while x_new[n]+var(x_new[n])/m/oversample*sampl<=x[-1]:
        x_new.append(x_new[n]+var(x_new[n])/m/oversample*sampl)
        n+=1
    x_new.append(x_new[n]+var(x_new[n])/m/oversample*sampl)
    if len(kernel(m*oversample/sampl))>len(x_new):
        raise RuntimeError('Kernel is larger than the data range.')
    y_new=np.interp(x_new,x,y)
    y_con=np.convolve(y_new,kernel(m*oversample/sampl),mode=mode)
    y_out=np.interp(x,x_new,y_con)
    return y_out
