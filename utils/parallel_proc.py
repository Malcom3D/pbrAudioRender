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
import numba as nb
import dask.array as da
from dask.distributed import Client, LocalCluster
from typing import Any, Callable

class ParallelProcessor:
    """Manages parallel processing configuration for CPU/GPU operations"""
    
    def __init__(self, system_config):
        self.system = system_config
        self.client = None
        self.cluster = None
        self._setup_parallel_environment()
    
    def _setup_parallel_environment(self):
        """Setup Dask cluster for for parallel processing"""
        if self.system.max_workers > 1:
            self.cluster = LocalCluster(
                n_workers=self.system.max_workers,
                threads_per_worker=1,
                processes=True
            )
            self.client = Client(self.cluster)
            print(f"Dask cluster started with {self.system.max_workers} workers")
    
    def parallel_map(self, func: Callable, arrays: list, **kwargs) -> list:
        """Apply function in parallel to multiple arrays"""
        if self.client is not None:
            # Use Dask for parallel execution
            futures = [self.client.submit(func, arr, **kwargs) for arr in arrays]
            return [future.result() for future in futures]
        else:
            # Sequential execution
            return [func(arr, **kwargs) for arr in arrays]
    
    def create_dask_array(self, numpy_array: np.ndarray, chunks: tuple = None) -> da.Array:
        """Convert numpy array to Dask array for parallel processing"""
        if chunks is None:
            # Auto-chunk based on array size
            chunks = self._auto_chunk(numpy_array.shape)
        
        return da.from_array(numpy_array, chunks=chunks)
    
    def _auto_chunk(self, shape: tuple) -> tuple:
        """Calculate optimal chunk size based on array shape"""
        chunks = []
        for dim in shape:
            if dim > 1024:
                chunks.append(256)
            elif dim > 512:
                chunks.append(128)
            else:
                chunks.append(dim)
        return tuple(chunks)
    
    def compute(self, dask_array: da.Array) -> np.ndarray:
        """Compute Dask array and return as numpy array"""
        return dask_array.compute()
    
    def close(self):
        """Clean up parallel processing resources"""
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()

# Common aliases for numpy, numba, and dask
def setup_array_backend(use_gpu: bool = False, compute_device: str = "cuda"):
    """Setup array computation backend based on configuration"""
    
    if use_gpu:
        try:
            if compute_device == "cuda":
                import cupy as xp
                print("Using CuPy (CUDA) backend")
            elif compute_device == "hip":
                import cupy as xp  
                print("Using CuPy (HIP) backend")
            else:
                raise ImportError(f"Unsupported GPU backend: {compute_device}")
        except ImportError:
            print("GPU backend not available, falling back to NumPy")
            import numpy as xp
    else:
        import numpy as xp
    
    return xp

# Numba configuration
def configure_numba(parallel: bool = True, fastmath: bool = True, cache: bool = True):
    """Configure Numba JIT compiler options"""
    nb.config.THREADING_LAYER = 'threadsafe'
    nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)
    
    def jit_decorator(func):
        return nb.jit(
            nopython=True,
            parallel=parallel,
            fastmath=fastmath,
            cache=cache
        )(func)
    
    return jit_decorator

