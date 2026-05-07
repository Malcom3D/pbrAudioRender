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

import os
import tempfile
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union

@dataclass
class AdaptiveArray:
    """
    Adaptive memory-mapped array with numpy compatibility.
    Handles accordion-like expansion/contraction patterns efficiently.
    Supports multidimensional arrays with axis-based operations.
    """
    memory_threshold: int = 1024 * 1024 * 100  # 100 MB default
    dtype: np.dtype = np.float32
    
    def __post_init__(self):
        self._chunks: List[np.ndarray] = []
        self._mmap: Optional[np.memmap] = None
        self._shape: Optional[List[int]] = None  # Track shape explicitly
        self._use_mmap = False
        self._mmap_path: Optional[str] = None
        self._total_size_along_axis0 = 0  # Track total size along axis 0
    
    def append(self, data: np.ndarray, axis: int = 0):
        """
        Append data along specified axis.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=self.dtype)
        elif data.dtype != self.dtype:
            data = data.astype(self.dtype)
        
        # Initialize shape if first append
        if self._shape is None:
            self._shape = list(data.shape)
        else:
            # Validate compatibility
            self._validate_append_compatibility(data, axis)
        
        # Decide whether to store in chunks or mmap
        estimated_size = (self._total_size_along_axis0 + data.shape[axis]) * np.dtype(self.dtype).itemsize
        if not self._use_mmap and estimated_size > self.memory_threshold:
            self._convert_to_mmap()
            # After conversion, append to mmap
            self._mmap_append(data, axis)
            return
        
        if not self._use_mmap:
            # Store in chunks
            self._chunks.append(data.copy())
            self._update_shape_after_append(data, axis)
            self._total_size_along_axis0 += data.shape[axis]
        else:
            self._mmap_append(data, axis)
    
    def _validate_append_compatibility(self, data: np.ndarray, axis: int):
        """Ensure data can be appended along axis."""
        if len(data.shape) != len(self._shape):
            raise ValueError(f"Data ndim {data.ndim} incompatible with current array ndim {len(self._shape)}")
        for i in range(len(self._shape)):
            if i != axis:
                if self._shape[i] != data.shape[i]:
                    raise ValueError(f"Shape mismatch at axis {i}: expected {self._shape[i]}, got {data.shape[i]}")
        # No shape check for axis dimension; appending along axis is allowed
    
    def _update_shape_after_append(self, data: np.ndarray, axis: int):
        """Update stored shape after appending data."""
        self._shape[axis] += data.shape[axis]
        # For other axes, shape remains the same
        
    def _convert_to_mmap(self):
        """Convert existing chunks to a memory-mapped file."""
        if self._shape is None:
            return
        # Create temp file
        fd, path = tempfile.mkstemp(suffix='.dat')
        os.close(fd)
        # Create mmap with current shape
        self._mmap = np.memmap(path, dtype=self.dtype, mode='w+', shape=tuple(self._shape))
        # Copy existing data
        if self._chunks:
            # Concatenate chunks along axis 0
            data = np.concatenate(self._chunks, axis=0)
            self._mmap[:] = data
        self._chunks = []
        self._use_mmap = True
        self._mmap_path = path
    
    def _mmap_append(self, data: np.ndarray, axis: int = 0):
        """Append data to memory-mapped array."""
        old_shape = tuple(self._shape)
        append_size = data.shape[axis]
        new_shape = list(self._shape)
        new_shape[axis] += append_size
        
        # Create new mmap with larger shape
        fd, path = tempfile.mkstemp(suffix='.dat')
        os.close(fd)
        new_mmap = np.memmap(path, dtype=self.dtype, mode='w+', shape=tuple(new_shape))
        
        # Copy old data into new mmap
        old_mmap = np.memmap(self._mmap_path, dtype=self.dtype, mode='r', shape=old_shape)
        if axis == 0:
            new_mmap[:old_shape[0]] = old_mmap[:]
            new_mmap[old_shape[0]:] = data
        else:
            # For other axes, handle slicing
            slicer_old = [slice(None)] * len(self._shape)
            slicer_new = [slice(None)] * len(self._shape)
            slicer_old[axis] = slice(0, old_shape[axis])
            slicer_new[axis] = slice(old_shape[axis], new_shape[axis])
            new_mmap[tuple(slicer_new)] = data
        # Close old mmap
        old_mmap._mmap.close()
        os.remove(self._mmap_path)
        self._mmap = new_mmap
        self._mmap_path = path
        self._shape = list(new_shape)
        self._total_size_along_axis0 += append_size
    
    @property
    def shape(self) -> Tuple[int, ...]:
        if self._shape is None:
            return (0,)
        return tuple(self._shape)
    
    @property
    def ndim(self) -> int:
        if self._shape is None:
            return 0
        return len(self._shape)

    @property 
    def size(self) -> int:
        """Get total number of elements"""
        if self._total_size == 0:
            return 0
        return np.prod(self._shape)

    def to_array(self) -> np.ndarray:
        """Reconstruct full array."""
        if self._use_mmap:
            return self._mmap[:]
        elif self._chunks:
            # Concatenate chunks along axis 0
            return np.concatenate(self._chunks, axis=0)
        else:
            # Empty array
            shape = tuple(self._shape) if self._shape else (0,)
            return np.empty(shape, dtype=self.dtype)
    
    def __getitem__(self, key):
        return self.to_array()[key]
    
    def __setitem__(self, key, value):
        """Set data at index."""
        arr = self.to_array()
        arr[key] = value
        # Replace with a new chunk
        self.clear()
        self.append(arr)
    
    def __len__(self):
        """Number of elements along axis 0."""
        if self._shape is None:
            return 0
        return self._shape[0]
    
    def __iter__(self):
        arr = self.to_array()
        for item in arr:
            yield item
    
    def filter(self, mask: np.ndarray, axis: int = 0) -> 'AdaptiveArray':
        data = self.to_array()
        filtered = np.compress(mask, data, axis=axis)
        result = AdaptiveArray(self.memory_threshold, self.dtype)
        result.append(filtered, axis=0)
        return result
    
    def clear(self):
        self._chunks = []
        if self._use_mmap and self._mmap is not None:
            try:
                self._mmap._mmap.close()
                os.remove(self._mmap_path)
            except:
                pass
        self._mmap = None
        self._shape = None
        self._total_size_along_axis0 = 0
        self._use_mmap = False
        self._mmap_path = None
    
    # Additional methods as needed (e.g., reshape, transpose, etc.)...
    # They should reconstruct data via to_array() and then create new AdaptiveArray.
    def reshape(self, *shape) -> 'AdaptiveArray':
        data = self.to_array().reshape(*shape)
        result = AdaptiveArray(self.memory_threshold, self.dtype)
        result.append(data)
        return result
    
    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'AdaptiveArray':
        data = self.to_array().transpose(axes)
        result = AdaptiveArray(self.memory_threshold, self.dtype)
        result.append(data)
        return result
    
    def squeeze(self, axis: Optional[int] = None) -> 'AdaptiveArray':
        data = self.to_array().squeeze(axis)
        result = AdaptiveArray(self.memory_threshold, self.dtype)
        result.append(data)
        return result
    
    def sum(self, axis: Optional[int] = None):
        return np.sum(self.to_array(), axis=axis)
    
    def mean(self, axis: Optional[int] = None):
        return np.mean(self.to_array(), axis=axis)
    
    def std(self, axis: Optional[int] = None):
        return np.std(self.to_array(), axis=axis)
    
    def min(self, axis: Optional[int] = None):
        return np.min(self.to_array(), axis=axis)
    
    def max(self, axis: Optional[int] = None):
        return np.max(self.to_array(), axis=axis)
    
    def argmax(self, axis: Optional[int] = None):
        return np.argmax(self.to_array(), axis=axis)
    
    def argmin(self, axis: Optional[int] = None):
        return np.argmin(self.to_array(), axis=axis)
    
    def copy(self) -> 'AdaptiveArray':
        data = self.to_array()
        result = AdaptiveArray(self.memory_threshold, self.dtype)
        result.append(data)
        return result
    
    def __del__(self):
        if self._use_mmap and self._mmap is not None:
            try:
                self._mmap._mmap.close()
                os.remove(self._mmap_path)
            except:
                pass
    
    def __repr__(self):
        return f"AdaptiveArray(shape={self.shape}, dtype={self.dtype}, use_mmap={self._use_mmap})"
    
    def __str__(self):
        return str(self.to_array())

    # Numpy compatibility methods
    def astype(self, dtype) -> 'AdaptiveArray':
        """Convert to different dtype"""
        data = self.to_array().astype(dtype)
        result = AdaptiveArray(self.memory_threshold, dtype)
        result.append(data)
        return result
   
    def flatten(self) -> 'AdaptiveArray':
        """Flatten array to 1D"""
        data = self.to_array().flatten()
        result = AdaptiveArray(self.memory_threshold, self.dtype)
        result.append(data)
        return result
   
    def ravel(self) -> 'AdaptiveArray':
        """Return flattened array"""
        return self.flatten()
   
    def sort(self, axis: int = -1, kind: str = 'quicksort'):
        """Sort the array in-place"""
        data = self.to_array()
        data.sort(axis=axis, kind=kind)
        self.clear()
        self.append(data)

    def unique(self) -> 'AdaptiveArray':
        """Return unique elements"""
        data = np.unique(self.to_array())
        result = AdaptiveArray(self.memory_threshold, self.dtype)
        result.append(data)
        return result

    def where(self, condition: np.ndarray, x: Any, y: Any) -> 'AdaptiveArray':
        """Return elements chosen from x or y depending on condition"""
        data = np.where(condition, x, y)
        result = AdaptiveArray(self.memory_threshold, self.dtype)
        result.append(data)
        return result

    def concatenate(self, others: List['AdaptiveArray'], axis: int = 0) -> 'AdaptiveArray':
        """Concatenate multiple AdaptiveArrays"""
        all_data = [self.to_array()]
        for other in others:
            all_data.append(other.to_array())

        concatenated = np.concatenate(all_data, axis=axis)
        result = AdaptiveArray(self.memory_threshold, self.dtype)
        result.append(concatenated)
        return result

    def split(self, indices_or_sections, axis: int = 0) -> List['AdaptiveArray']:
        """Split array into multiple sub-arrays"""
        data = self.to_array()
        splits = np.split(data, indices_or_sections, axis=axis)

        results = []
        for split_data in splits:
            result = AdaptiveArray(self.memory_threshold, self.dtype)
            result.append(split_data)
            results.append(result)

        return results

    def pad(self, pad_width, mode: str = 'constant', **kwargs) -> 'AdaptiveArray':
        """Pad the array"""
        data = self.to_array()
        padded = np.pad(data, pad_width, mode=mode, **kwargs)

        result = AdaptiveArray(self.memory_threshold, self.dtype)
        result.append(padded)
        return result

    def clip(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> 'AdaptiveArray':
        """Clip values to a range"""
        data = np.clip(self.to_array(), min_val, max_val)
        result = AdaptiveArray(self.memory_threshold, self.dtype)
        result.append(data)
        return result

    def fill(self, value: Any):
        """Fill array with a scalar value"""
        if self._use_mmap:
            self._mmap.fill(value)
        else:
            for chunk in self._chunks:
                chunk.fill(value)

    def nonzero(self) -> Tuple[np.ndarray, ...]:
        """Return indices of non-zero elements"""
        return np.nonzero(self.to_array())

    def all(self, axis: Optional[int] = None) -> Union[bool, np.ndarray]:
        """Test whether all elements evaluate to True"""
        return np.all(self.to_array(), axis=axis)

    def any(self, axis: Optional[int] = None) -> Union[bool, np.ndarray]:
        """Test whether any elements evaluate to True"""
        return np.any(self.to_array(), axis=axis)

