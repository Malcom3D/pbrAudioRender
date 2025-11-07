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
import os
from typing import Dict, List, Tuple, Optional, Any
from pympler import asizeof

from core.entity_manager import EntityManager

from lib.acoustic_field import VelocityVectors, FrequencyLimitedField, AcousticField


class GPUManager:
    """Manages GPU devices and memory"""
    
    def __init__(self, entity_manager: EntityManager):
        self.config = entity_manager.get('config')
        self.device = None
        self.context = None
        self.streams = []
        self.memory_allocated = 0
        self.memory_peak = 0
        
        if self.config.gpu.use_gpu:
            self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Initialize GPU based on selected backend"""
        try:
            if self.config.gpu.compute_device == "cuda":
                self._initialize_cuda()
            elif self.config.gpu.compute_device == "hip":
                self._initialize_hip()
            elif self.config.gpu.compute_device == "oneapi":
                self._initialize_oneapi()
            elif self.config.gpu.compute_device == "opencl":
                self._initialize_opencl()
            else:
                print(f"Warning: Unsupported GPU backend: {self.config.gpu.compute_device}")
                self.config.gpu.use_gpu = False
                
        except Exception as e:
            print(f"Warning: GPU initialization failed: {e}. Falling back to CPU.")
            self.config.gpu.use_gpu = False
    
    def _initialize_cuda(self):
        """Initialize CUDA backend"""
        try:
            import cupy as cp
            from cupy.cuda import Device, Stream
            
            self.backend = 'cuda'
            self.cp = cp
            self.Device = Device
            self.Stream = Stream
            
            # Set device
            self.Device(self.config.gpu.device_id).use()
            self.device = self.Device(self.config.gpu.device_id)
            
            # Create streams
            for i in range(self.config.gpu.num_streams):
                self.streams.append(Stream(non_blocking=True))
                
            print(f"CUDA initialized on device {self.config.gpu.device_id}: {self.device.name}")
            
        except ImportError:
            raise ImportError("CuPy not available. Install with: pip install cupy-cuda11x")
    
    def _initialize_hip(self):
        """Initialize HIP backend (AMD GPUs)"""
        try:
            # HIP uses the same CuPy interface but with ROCm
            import cupy as cp
            from cupy.cuda import Device, Stream
            
            self.backend = 'hip'
            self.cp = cp
            self.Device = Device
            self.Stream = Stream
            
            self.Device(self.config.gpu.device_id).use()
            self.device = self.Device(self.config.gpu.device_id)
            
            for i in range(self.config.gpu.num_streams):
                self.streams.append(Stream(non_blocking=True))
                
            print(f"HIP initialized on device {self.config.gpu.device_id}: {self.device.name}")
            
        except ImportError:
            raise ImportError("CuPy with HIP support not available.")
    
    def _initialize_oneapi(self):
        """Initialize oneAPI backend (Intel GPUs)"""
        try:
            import dpctl
            import dpctl.tensor as dpt
            from dpctl.stream import Stream
            
            self.backend = 'oneapi'
            self.dpctl = dpctl
            self.dpt = dpt
            
            # Select device
            devices = dpctl.get_devices(backend='level_zero')
            if devices and self.config.gpu.device_id < len(devices):
                self.device = devices[self.config.gpu.device_id]
            else:
                self.device = dpctl.select_default_device()
            
            # Create streams
            for i in range(self.config.gpu.num_streams):
                self.streams.append(Stream(self.device))
                
            print(f"oneAPI initialized on device: {self.device.name}")
            
        except ImportError:
            raise ImportError("dpctl not available. Install with: pip install dpctl")
    
    def _initialize_opencl(self):
        """Initialize OpenCL backend (multi-vendor)"""
        try:
            import pyopencl as cl
            import pyopencl.array as cl_array
            
            self.backend = 'opencl'
            self.cl = cl
            self.cl_array = cl_array
            
            # Get platforms and devices
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found")
            
            # Select device
            devices = []
            for platform in platforms:
                devices.extend(platform.get_devices())
            
            if self.config.gpu.device_id < len(devices):
                self.device = devices[self.config.gpu.device_id]
            else:
                self.device = devices[0]
            
            # Create context and command queue
            self.context = cl.Context([self.device])
            self.streams = [cl.CommandQueue(self.context) for _ in range(self.config.gpu.num_streams)]
            
            print(f"OpenCL initialized on device: {self.device.name}")
            
        except ImportError:
            raise ImportError("PyOpenCL not available. Install with: pip install pyopencl")
    
    def allocate_memory(self, shape: Tuple[int, ...]) -> Any:
        """Allocate GPU memory"""
        if not self.config.gpu.use_gpu:
            return np.zeros(shape, dtype=object)
        
        # Get total size of one fully allocated AcousticField:
        low_freq = self.config.fdtd.lowest_frequency
        high_freq = self.config.acoustic_domain.sample_rate / 2
        bands_per_octave = self.config.acoustic_domain.bands_per_octave
        num_bands_freq = len(em.frequencies)
        vel = VelocityVectors(x=1.234, y=4.321, z=0.987)
        acoustic_field = AcousticField()
        for x in range(num_bands_freq):
            acoustic_field.add_field(low_freq=low_freq, high_freq=high_freq, pressure=0.123, velocity=vel)
        nbytes = asizeof.asizeof(acoustic_field)
        
        # Check memory limits
        if (self.config.gpu.memory_limit and 
            self.memory_allocated + nbytes > self.config.gpu.memory_limit * 1024 * 1024):
            raise MemoryError(f"GPU memory limit exceeded: {self.memory_allocated / 1024 / 1024:.1f}MB")
        
        try:
            if self.backend in ['cuda', 'hip']:
                array = self.cp.zeros(shape, dtype=dtype)
            elif self.backend == 'oneapi':
                array = self.dpt.zeros(shape, dtype=dtype, device=self.device)
            elif self.backend == 'opencl':
                array = self.cl_array.zeros(self.streams[0], shape, dtype)
            else:
                return np.zeros(shape, dtype=dtype)
            
            self.memory_allocated += nbytes
            self.memory_peak = max(self.memory_peak, self.memory_allocated)
            return array
            
        except Exception as e:
            print(f"Warning: GPU allocation failed: {e}. Falling back to CPU.")
            return np.zeros(shape, dtype=dtype)
    
    def to_gpu(self, numpy_array: np.ndarray, stream_id: int = 0) -> Any:
        """Transfer numpy array to GPU"""
        if not self.config.gpu.use_gpu or numpy_array is None:
            return numpy_array
        
        try:
            if self.backend in ['cuda', 'hip']:
                with self.Stream(self.streams[stream_id]):
                    return self.cp.asarray(numpy_array)
            elif self.backend == 'oneapi':
                return self.dpt.asarray(numpy_array, device=self.device)
            elif self.backend == 'opencl':
                return self.cl_array.to_device(self.streams[stream_id], numpy_array)
                
        except Exception as e:
            print(f"Warning: GPU transfer failed: {e}. Using CPU array.")
            return numpy_array
    
    def to_cpu(self, gpu_array: Any, stream_id: int = 0) -> np.ndarray:
        """Transfer GPU array back to CPU"""
        if not self.config.gpu.use_gpu or gpu_array is None:
            return gpu_array
        
        try:
            if self.backend in ['cuda', 'hip']:
                with self.Stream(self.streams[stream_id]):
                    return self.cp.asnumpy(gpu_array)
            elif self.backend == 'oneapi':
                return np.asarray(gpu_array)
            elif self.backend == 'opencl':
                return gpu_array.get()
                
        except Exception as e:
            print(f"Warning: GPU to CPU transfer failed: {e}")
            return np.array(gpu_array) if hasattr(gpu_array, '__array__') else None
    
    def synchronize(self, stream_id: int = None):
        """Synchronize GPU operations"""
        if not self.config.gpu.use_gpu:
            return
        
        try:
            if stream_id is not None:
                if self.backend in ['cuda', 'hip']:
                    self.streams[stream_id].synchronize()
                elif self.backend == 'oneapi':
                    self.streams[stream_id].wait()
                elif self.backend == 'opencl':
                    self.streams[stream_id].finish()
            else:
                for stream in self.streams:
                    if self.backend in ['cuda', 'hip']:
                        stream.synchronize()
                    elif self.backend == 'oneapi':
                        stream.wait()
                    elif self.backend == 'opencl':
                        stream.finish()
                        
        except Exception as e:
            print(f"Warning: GPU synchronization failed: {e}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information"""
        if not self.config.gpu.use_gpu:
            return {'allocated': 0, 'peak': 0}
        
        try:
            if self.backend in ['cuda', 'hip']:
                free, total = self.cp.cuda.runtime.memGetInfo()
                return {
                    'allocated': self.memory_allocated / 1024 / 1024,
                    'peak': self.memory_peak / 1024 / 1024,
                    'free': free / 1024 / 1024,
                    'total': total / 1024 / 1024
                }
            else:
                return {
                    'allocated': self.memory_allocated / 1024 / 1024,
                    'peak': self.memory_peak / 1024 / 1024
                }
        except:
            return {'allocated': 0, 'peak': 0}

class GPUArrayManager:
    """Manages GPU arrays for acoustic simulation"""
    
    def __init__(self, gpu_manager: GPUManager, grid_shape: Tuple[int, int, int]):
        self.gpu = gpu_manager
        self.shape = grid_shape
        self.arrays = {}
        
    def register_array(self, name: str, dtype: np.dtype = np.float32):
        """Register a new GPU array"""
        self.arrays[name] = self.gpu.allocate_memory(self.shape, dtype)
        return self.arrays[name]
    
    def get_array(self, name: str) -> Any:
        """Get GPU array by name"""
        return self.arrays.get(name)
    
    def update_from_cpu(self, name: str, cpu_array: np.ndarray, stream_id: int = 0):
        """Update GPU array from CPU data"""
        if name in self.arrays:
            gpu_array = self.gpu.to_gpu(cpu_array, stream_id)
            if self.gpu.config.gpu.use_gpu:
                self.arrays[name][:] = gpu_array
            else:
                self.arrays[name] = gpu_array
    
    def get_to_cpu(self, name: str, stream_id: int = 0) -> np.ndarray:
        """Get GPU array as CPU numpy array"""
        if name in self.arrays:
            return self.gpu.to_cpu(self.arrays[name], stream_id)
        return None
    
    def sync_all(self):
        """Synchronize all GPU operations"""
        self.gpu.synchronize()

