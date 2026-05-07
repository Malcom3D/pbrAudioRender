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
        self._total_size = 0
        self._use_mmap = False
        self._shape = None
        self._ndim = 0
        
    def append(self, data: np.ndarray, axis: int = 0):
        """
        Append data along specified axis.
        
        Args:
            data: Array to append
            axis: Axis along which to append (default: 0)
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=self.dtype)
        
        # Ensure data has correct dtype
        if data.dtype != self.dtype:
            data = data.astype(self.dtype)
        
        # Initialize shape if first append
        if self._total_size == 0:
            self._shape = list(data.shape)
            self._ndim = data.ndim
        else:
            # Validate compatibility
            self._validate_append_compatibility(data, axis)
        
        if not self._use_mmap:
            # Check if we should switch to mmap
            estimated_size = (self._total_size + self._get_append_size(data, axis)) * \
                           np.dtype(self.dtype).itemsize
            if estimated_size > self.memory_threshold:
                self._convert_to_mmap()
                self.append(data, axis)
                return
            
            self._chunks.append(data.copy())
            self._total_size += self._get_append_size(data, axis)
            self._update_shape_after_append(data, axis)
        else:
            # Append to mmap
            self._mmap_append(data, axis)
    
    def _get_append_size(self, data: np.ndarray, axis: int) -> int:
        """Get the size contribution of data along the append axis"""
        return data.shape[axis]
    
    def _validate_append_compatibility(self, data: np.ndarray, axis: int):
        """Validate that data can be appended along the given axis"""
        if data.ndim != self._ndim:
            raise ValueError(f"Cannot append {data.ndim}D data to {self._ndim}D array")
        
        for i in range(data.ndim):
            if i != axis:
                if data.shape[i] != self._shape[i]:
                    raise ValueError(
                        f"Shape mismatch at axis {i}: expected {self._shape[i]}, "
                        f"got {data.shape[i]}"
                    )
    
    def _update_shape_after_append(self, data: np.ndarray, axis: int):
        """Update the stored shape after appending"""
        self._shape[axis] += data.shape[axis]
    
    def _convert_to_mmap(self):
        """Convert existing chunks to memory-mapped file"""
        if self._total_size == 0:
            return
        
        # Create temp file
        fd, path = tempfile.mkstemp(suffix='.dat')
        os.close(fd)
        
        # Create mmap with current data and shape
        mmap_shape = tuple(self._shape)
        self._mmap = np.memmap(path, dtype=self.dtype, mode='w+', shape=mmap_shape)
        
        # Copy existing data
        if self._chunks:
            pos = 0
            for chunk in self._chunks:
                if self._ndim == 1:
                    self._mmap[pos:pos + len(chunk)] = chunk
                else:
                    # For multidimensional, we need to handle concatenation properly
                    if pos == 0:
                        self._mmap[:chunk.shape[0]] = chunk
                    else:
                        self._mmap[pos:pos + chunk.shape[0]] = chunk
                pos += chunk.shape[0] if self._ndim == 1 else chunk.shape[0]
        
        self._chunks = None  # Free memory
        self._use_mmap = True
        self._mmap_path = path
    
    def _mmap_append(self, data: np.ndarray, axis: int = 0):
        """Append data to memory-mapped file"""
        old_size = self._total_size
        append_size = self._get_append_size(data, axis)
        new_size = old_size + append_size
        
        # Update shape
        old_shape = list(self._shape)
        self._shape[axis] = new_size
        
        # Create new mmap with updated shape
        new_shape = tuple(self._shape)
        
        # We need to resize the mmap
        self._mmap._mmap.close()
        self._mmap = None
        
        # Create new mmap file
        fd, path = tempfile.mkstemp(suffix='.dat')
        os.close(path)
        
        new_mmap = np.memmap(path, dtype=self.dtype, mode='w+', shape=new_shape)
        
        # Copy old data
        if old_size > 0:
            # Read old mmap
            old_path = self._mmap_path
            old_mmap = np.memmap(old_path, dtype=self.dtype, mode='r', shape=old_shape)
            
            # Copy to new mmap
            if axis == 0:
                new_mmap[:old_shape[0]] = old_mmap[:]
            else:
                # For other axes, we need to handle differently
                slicing = [slice(None)] * self._ndim
                slicing[axis] = slice(0, old_shape[axis])
                new_mmap[tuple(slicing)] = old_mmap[:]
            
            old_mmap._mmap.close()
            os.remove(old_path)
        
        # Append new data
        if axis == 0:
            new_mmap[old_shape[0]:new_shape[0]] = data
        else:
            slicing = [slice(None)] * self._ndim
            slicing[axis] = slice(old_shape[axis], new_shape[axis])
            new_mmap[tuple(slicing)] = data
        
        self._mmap = new_mmap
        self._mmap_path = path
        self._total_size = new_size
    
    def extend(self, data: Union[np.ndarray, List], axis: int = 0):
        """Alias for append"""
        self.append(data, axis)
    
    def filter(self, mask: np.ndarray, axis: int = 0) -> 'AdaptiveArray':
        """
        Apply boolean filter along specified axis.
        
        Args:
            mask: Boolean mask array
            axis: Axis along which to filter
            
        Returns:
            New AdaptiveArray with filtered data
        """
        data = self.to_array()
        
        if mask.ndim == 1 and axis == 0:
            filtered = data[mask]
        else:
            # For multidimensional filtering along arbitrary axis
            if mask.ndim != data.ndim:
                # Expand mask dimensions if needed
                while mask.ndim < data.ndim:
                    mask = np.expand_dims(mask, axis=-1)
            
            # Apply mask along specified axis
            indices = np.where(mask)[axis]
            filtered = np.take(data, indices, axis=axis)
        
        result = AdaptiveArray(self.memory_threshold, self.dtype)
        result.append(filtered, axis=0)
        return result
    
    def to_array(self) -> np.ndarray:
        """Convert to regular numpy array"""
        if self._total_size == 0:
            return np.array([], dtype=self.dtype)
        
        if self._use_mmap:
            return self._mmap[:].copy()
        else:
            if len(self._chunks) == 1:
                return self._chunks[0].copy()
            elif len(self._chunks) > 1:
                for i in range(len(self._chunks)):
                    print(self._chunks[i].shape)
                return np.concatenate(self._chunks, axis=0)
            return np.array([], dtype=self.dtype)
    
    def __getitem__(self, key):
        """Support numpy-like indexing"""
        data = self.to_array()
        return data[key]
    
    def __setitem__(self, key, value):
        """Support numpy-like assignment"""
        if self._use_mmap:
            self._mmap[key] = value
        else:
            # For chunk-based storage, we need to modify the appropriate chunk
            # This is simplified - for production, you'd need more complex logic
            data = self.to_array()
            data[key] = value
            self._chunks = [data]
            self._total_size = len(data)
    
    def __len__(self) -> int:
        """Return length along first axis"""
        return self._total_size if self._ndim <= 1 else self._shape[0]
    
    def __iter__(self):
        """Iterate over first axis"""
        for i in range(len(self)):
            yield self[i]
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get array shape"""
        if self._total_size == 0:
            return (0,)
        return tuple(self._shape)
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions"""
        return self._ndim
    
    @property
    def size(self) -> int:
        """Get total number of elements"""
        if self._total_size == 0:
            return 0
        return np.prod(self._shape)
    
    def reshape(self, *shape) -> 'AdaptiveArray':
        """Reshape the array (creates a new AdaptiveArray)"""
        data = self.to_array()
        reshaped = data.reshape(*shape)
        
        result = AdaptiveArray(self.memory_threshold, self.dtype)
        result.append(reshaped)
        return result
    
    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'AdaptiveArray':
        """Transpose the array (creates a new AdaptiveArray)"""
        data = self.to_array()
        transposed = data.transpose(axes)
        
        result = AdaptiveArray(self.memory_threshold, self.dtype)
        result.append(transposed)
        return result
    
    def squeeze(self, axis: Optional[int] = None) -> 'AdaptiveArray':
        """Remove single-dimensional entries from shape"""
        data = self.to_array()
        squeezed = np.squeeze(data, axis=axis)
        
        result = AdaptiveArray(self.memory_threshold, self.dtype)
        result.append(squeezed)
        return result
    
    def sum(self, axis: Optional[int] = None) -> Union[np.ndarray, float]:
        """Sum along specified axis"""
        data = self.to_array()
        return np.sum(data, axis=axis)
    
    def mean(self, axis: Optional[int] = None) -> Union[np.ndarray, float]:
        """Mean along specified axis"""
        data = self.to_array()
        return np.mean(data, axis=axis)
    
    def std(self, axis: Optional[int] = None) -> Union[np.ndarray, float]:
        """Standard deviation along specified axis"""
        data = self.to_array()
        return np.std(data, axis=axis)
    
    def min(self, axis: Optional[int] = None) -> Union[np.ndarray, float]:
        """Minimum along specified axis"""
        data = self.to_array()
        return np.min(data, axis=axis)
    
    def max(self, axis: Optional[int] = None) -> Union[np.ndarray, float]:
        """Maximum along specified axis"""
        data = self.to_array()
        return np.max(data, axis=axis)
    
    def argmax(self, axis: Optional[int] = None) -> Union[np.ndarray, int]:
        """Indices of maximum values along axis"""
        data = self.to_array()
        return np.argmax(data, axis=axis)
    
    def argmin(self, axis: Optional[int] = None) -> Union[np.ndarray, int]:
        """Indices of minimum values along axis"""
        data = self.to_array()
        return np.argmin(data, axis=axis)
    
    def copy(self) -> 'AdaptiveArray':
        """Create a deep copy"""
        result = AdaptiveArray(self.memory_threshold, self.dtype)
        data = self.to_array()
        result.append(data)
        return result
    
    def clear(self):
        """Clear all data"""
        self._chunks = []
        if self._use_mmap and hasattr(self, '_mmap_path'):
            try:
                self._mmap._mmap.close()
                os.remove(self._mmap_path)
            except:
                pass
        self._mmap = None
        self._total_size = 0
        self._use_mmap = False
        self._shape = None
        self._ndim = 0
    
    def __del__(self):
        """Cleanup memory-mapped files"""
        if self._use_mmap and hasattr(self, '_mmap_path'):
            try:
                self._mmap._mmap.close()
                os.remove(self._mmap_path)
            except:
                pass
    
    def __repr__(self) -> str:
        return f"AdaptiveArray(shape={self.shape}, dtype={self.dtype}, use_mmap={self._use_mmap})"
    
    def __str__(self) -> str:
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
