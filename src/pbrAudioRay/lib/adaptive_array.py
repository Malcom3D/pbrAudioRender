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
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import mmap
import queue
from contextlib import contextmanager

@dataclass
class AdaptiveArray:
    """
    Adaptive memory-mapped array with numpy compatibility.
    Handles accordion-like expansion/contraction patterns efficiently.
    Supports multidimensional arrays with axis-based operations.
    Parallel I/O for memory-mapped operations.
    """
    memory_threshold: int = 1024 * 1024 * 100  # 100 MB default
    dtype: np.dtype = np.float32
    num_io_workers: int = 4  # Number of parallel I/O workers
    
    def __post_init__(self):
        self._chunks: List[np.ndarray] = []
        self._mmap: Optional[np.memmap] = None
        self._total_size = 0
        self._use_mmap = False
        self._shape = None
        self._ndim = 0
        self._lock = Lock()  # Thread safety for state changes
        self._io_executor = ThreadPoolExecutor(max_workers=self.num_io_workers)
        self._io_queue = queue.Queue()
        self._pending_io = []
        self._mmap_path = None
        
    def append(self, data: np.ndarray, axis: int = 0):
        """
        Append data along specified axis with parallel I/O support.
        
        Args:
            data: Array to append
            axis: Axis along which to append (default: 0)
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=self.dtype)
        
        # Ensure data has correct dtype
        if data.dtype != self.dtype:
            data = data.astype(self.dtype)
        
        with self._lock:
            # Initialize shape if first append
            if self._total_size == 0:
                self._shape = list(data.shape)
                self._ndim = data.ndim
            else:
                # Validate compatibility
                self._validate_append_compatibility(data, axis)
            
            if not self._use_mmap:
                # Check if we should switch to mmap
                estimated_size = (self._total_size + self._get_append_size(data, axis)) * np.dtype(self.dtype).itemsize
                if estimated_size > self.memory_threshold:
                    self._convert_to_mmap()
                    # Re-append after conversion
                    self._mmap_append(data, axis)
                    return
                
                self._chunks.append(data.copy())
                self._total_size += self._get_append_size(data, axis)
                self._update_shape_after_append(data, axis)
            else:
                # Parallel mmap append
                self._mmap_append_parallel(data, axis)
    
    def _get_append_size(self, data: np.ndarray, axis: int) -> int:
        """Get the size contribution of data along the append axis"""
        return data.shape[axis]
    
    def _validate_append_compatibility(self, data: np.ndarray, axis: int):
        """Validate that data can be appended along the given axis"""
        if data.ndim != self._ndim:
            raise ValueError(f"Cannot append {data.ndim}D data to {self._ndim}D array")
        
        for i in range(data.ndim):
            if i != axis:
                if self._shape is not None and data.shape[i] != self._shape[i]:
                    raise ValueError(
                        f"Shape mismatch at axis {i}: expected {self._shape[i]}, "
                        f"got {data.shape[i]}"
                    )
    
    def _update_shape_after_append(self, data: np.ndarray, axis: int):
        """Update the stored shape after appending along axis."""
        if self._shape is None:
            self._shape = list(data.shape)
            self._ndim = data.ndim
        else:
            # Increase only along the appended axis
            self._shape[axis] += data.shape[axis]
            # For other axes, ensure shape compatibility
            for i in range(self._ndim):
                if i != axis:
                    if self._shape[i] != data.shape[i]:
                        raise ValueError(f"Shape mismatch at axis {i} after append: expected {self._shape[i]}, got {data.shape[i]}")
    
    def _convert_to_mmap(self):
        """Convert existing chunks to memory-mapped file with parallel I/O"""
        if self._total_size == 0:
            return
        
        # Create temp file
        fd, path = tempfile.mkstemp(suffix='.dat')
        os.close(fd)
        
        # Create mmap with current data and shape
        mmap_shape = tuple(self._shape)
        self._mmap = np.memmap(path, dtype=self.dtype, mode='w+', shape=mmap_shape)
        
        # Copy existing data in parallel
        if self._chunks:
            futures = []
            chunk_size = max(1, len(self._chunks) // self.num_io_workers)
            
            for i in range(0, len(self._chunks), chunk_size):
                chunk_batch = self._chunks[i:i+chunk_size]
                start_pos = sum(c.shape[0] for c in self._chunks[:i])
                future = self._io_executorutor.submit(
                    self._parallel_chunk_copy,
                    chunk_batch, start_pos, mmap_shape
                )
                futures.append(future)
            
            # Wait for all parallel copies to complete
            for future in as_completed(futures):
                future.result()  # Propagate any exceptions
        
        self._chunks = None  # Free memory
        self._use_mmap = True
        self._mmap_path = path
    
    def _parallel_chunk_copy(self, chunks: List[np.ndarray], start_pos: int, shape: Tuple):
        """Copy chunks to mmap in parallel"""
        # Create local mmap reference for this thread
        local_mmap = np.memmap(self._mmap_path, dtype=self.dtype, mode='r+', shape=shape)
        
        pos = start_pos
        for chunk in chunks:
            if self._ndim == 1:
                local_mmap[pos:pos + len(chunk)] = chunk
            else:
                local_mmap[pos:pos + chunk.shape[0]] = chunk
            pos += chunk.shape[0]
        
        # Flush and close
        local_mmap.flush()
        del local_mmap
    
    def _mmap_append_parallel(self, data: np.ndarray, axis: int = 0):
        """Append data to memory-mapped file using parallel I/O"""
        old_size = self._total_size
        append_size = self._get_append_size(data, axis)
        new_size = old_size + append_size
        
        # Update shape
        old_shape = list(self._shape)
        self._shape[axis] = new_size
        
        # Create new mmap with updated shape
        new_shape = tuple(self._shape)
        
        # Close old mmap
        if self._mmap is not None:
            self._mmap._mmap.close()
            self._mmap = None
        
        # Create new mmap file
        fd, new_path = tempfile.mkstemp(suffix='.dat')
        os.close(fd)
        
        new_mmap = np.memmap(new_path, dtype=self.dtype, mode='w+', shape=new_shape)
        
        # Use parallel I/O for copying and appending
        if old_size > 0:
            # Read old data in parallel
            old_path = self._mmap_path
            old_mmap = np.memmap(old_path, dtype=self.dtype, mode='r', shape=old_shape)
            
            # Split the copy operation into chunks for parallel processing
            chunk_size = max(1, old_shape[0] // self.num_io_workers)
            futures = []
            
            for start in range(0, old_shape[0], chunk_size):
                end = min(start + chunk_size, old_shape[0])
                future = self._io_executor.submit(
                    self._parallel_copy_slice,
                    old_path, new_path, start, end, old_shape, new_shape, axis
                )
                futures.append(future)
            
            # Wait for all copies to complete
            for future in as_completed(futures):
                future.result()
            
            old_mmap._mmap.close()
            os.remove(old_path)
        
        # Append new data in parallel if large enough
        if data.size > 1000000:  # 1MB threshold for parallel append
            self._parallel_append_data(data, new_mmap, old_shape, new_shape, axis)
        else:
            # Sequential append for small data
            if axis == 0:
                new_mmap[old_shape[0]:new_shape[0]] = data
            else:
                slicing = [slice(None)] * self._ndim
                slicing[axis] = slice(old_shape[axis], new_shape[axis])
                new_mmap[tuple(slicing)] = data
        
        self._mmap = new_mmap
        self._mmap_path = new_path
        self._total_size = new_size
    
    def _parallel_copy_slice(self, old_path: str, new_path: str, 
                            start: int, end: int, old_shape: Tuple, 
                            new_shape: Tuple, axis: int):
        """Copy a slice of data between mmap files in parallel"""
        # Open both mmaps in this thread
        old_local = np.memmap(old_path, dtype=self.dtype, mode='r', shape=old_shape)
        new_local = np.memmap(new_path, dtype=self.dtype, mode='r+', shape=new_shape)
        
        # Copy the slice
        if axis == 0:
            new_local[start:end] = old_local[start:end]
        else:
            slicing_old = [slice(None)] * self._ndim
            slicing_old[axis] = slice(start, end)
            slicing_new = [slice(None)] * self._ndim
            slicing_new[axis] = slice(start, end)
            new_local[tuple(slicing_new)] = old_local[tuple(slicing_old)]
        
        # Flush and close
        new_local.flush()
        del new_local
        del old_local
    
    def _parallel_append_data(self, data: np.ndarray, mmap: np.memmap,
                             old_shape: List, new_shape: List, axis: int):
        """Append data in parallel chunks"""
        chunk_size = max(1, data.shape[axis] // self.num_io_workers)
        futures = []
        
        for start in range(0, data.shape[axis], chunk_size):
            end = min(start + chunk_size, data.shape[axis])
            future = self._io_executor.submit(
                self._parallel_append_chunk,
                data, mmap, start, end, old_shape, axis
            )
            futures.append(future)
        
        for future in as_completed(futures):
            future.result()
    
    def _parallel_append_chunk(self, data: np.ndarray, mmap: np.memmap,
                              start: int, end: int, old_shape: List, axis: int):
        """Append a chunk of data to mmap"""
        # Get slice indices
        if axis == 0:
            old_size = old_shape[0] if old_shape else 0
            mmap[old_size + start:old_size + end] = data[start:end]
        else:
            old_size = old_shape[axis]
            slicing = [slice(None)] * self._ndim
            slicing[axis] = slice(old_size + start, old_size + end)
            data_slicing = [slice(None)] * self._ndim
            data_slicing[axis] = slice(start, end)
            mmap[tuple(slicing)] = data[tuple(data_slicing)]
        
        mmap.flush()
    
    def extend(self, data: Union[np.ndarray, List], axis: int = 0):
        """Alias for append"""
        self.append(data, axis)
    
    def filter(self, mask: np.ndarray, axis: int = 0) -> 'AdaptiveArray':
        """
        Apply boolean filter along specified axis with parallel processing.
        
        Args:
            mask: Boolean mask array
            axis: Axis along which to filter
            
        Returns:
            New AdaptiveArray with filtered data
        """
        data = self.to_array()
        
        if mask.ndim == 1 and axis == 0:
            # Parallel filtering
            chunk_size = max(1, len(data) // self.num_io_workers)
            chunks = []
            
            with ThreadPoolExecutor(max_workers=self.num_io_workers) as executor:
                futures = []
                for i in range(0, len(data), chunk_size):
                    end = min(i + chunk_size, len(data))
                    future = executor.submit(
                        self._parallel_filter_chunk,
                        data[i:end], mask[i:end]
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    chunk_result = future.result()
                    if chunk_result is not None and len(chunk_result) > 0:
                        chunks.append(chunk_result)
            
            if chunks:
                filtered = np.concatenate(chunks)
            else:
                filtered = np.array([], dtype=self.dtype)
        else:
            # For multidimensional filtering along arbitrary axis
            if mask.ndim != data.ndim:
                while mask.ndim < data.ndim:
                    mask = np.expand_dims(mask, axis=-1)
            
            indices = np.where(mask)[axis]
            filtered = np.take(data, indices, axis=axis)
        
        result = AdaptiveArray(self.memory_threshold, self.dtype, self.num_io_workers)
        result.append(filtered, axis=0)
        return result
    
    def _parallel_filter_chunk(self, data_chunk: np.ndarray, mask_chunk: np.ndarray) -> np.ndarray:
        """Filter a chunk of data in parallel"""
        if mask_chunk.any():
            return data_chunk[mask_chunk]
        return None
    
    def to_array(self) -> np.ndarray:
        """Convert to regular numpy array with robust handling of chunks."""
        if self._total_size == 0:
            return np.array([], dtype=self.dtype)
    
        if self._use_mmap:
            return self._mmap[:].copy()
        else:
            if self._chunks:
                # Filter out empty chunks
                non_empty_chunks = [chunk for chunk in self._chunks if chunk.size > 0]
                if len(non_empty_chunks) == 0:
                    shape = list(self._shape) if self._shape is not None else (0,)
                    return np.empty(shape, dtype=self.dtype)
                elif len(non_empty_chunks) == 1:
                    return non_empty_chunks[0].copy()
                else:
                    # Parallel concatenation for large arrays
                    total_size = sum(chunk.shape[0] for chunk in non_empty_chunks)
                    if total_size > 1000000:  # 1MB threshold
                        return self._parallel_concat(non_empty_chunks)
                    return np.concatenate(non_empty_chunks, axis=0)
            else:
                return np.array([], dtype=self.dtype)
    
    def _parallel_concat(self, chunks: List[np.ndarray]) -> np.ndarray:
        """Concatenate chunks in parallel"""
        # Calculate total size
        total_size = sum(chunk.shape[0] for chunk in chunks)
        
        # Create output array
        result = np.empty((total_size,) + chunks[0].shape[1:], dtype=self.dtype)
        
        # Copy chunks in parallel
        with ThreadPoolExecutor(max_workers=self.num_io_workers) as executor:
            futures = []
            start = 0
            for chunk in chunks:
                end = start + chunk.shape[0]
                future = executor.submit(
                    self._parallel_copy_chunk,
                    result, chunk, start
                )
                futures.append(future)
                start = end
            
            for future in as_completed(futures):
                future.result()
        
        return result
    
    def _parallel_copy_chunk(self, dest: np.ndarray, src: np.ndarray, start: int):
        """Copy a chunk to destination array"""
        dest[start:start + src.shape[0]] = src
    
    def __getitem__(self, key):
        """Support numpy-like indexing"""
        data = self.to_array()
        return data[key]
    
    def __setitem__(self, key, value):
        """Support numpy-like assignment"""
        if self._use_mmap:
            self._mmap[key] = value
        else:
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
        
        result = AdaptiveArray(self.memory_threshold, self.dtype, self.num_io_workers)
        result.append(reshaped)
        return result
    
    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'AdaptiveArray':
        """Transpose the array (creates a new AdaptiveArray)"""
        data = self.to_array()
        transposed = data.transpose(axes)
        
        result = AdaptiveArray(self.memory_threshold, self.dtype, self.num_io_workers)
        result.append(transposed)
        return result
    
    def squeeze(self, axis: Optional[int] = None) -> 'AdaptiveArray':
        """Remove single-dimensional entries from shape"""
        data = self.to_array()
        squeezed = np.squeeze(data, axis=axis)
        
        result = AdaptiveArray(self.memory_threshold, self.dtype, self.num_io_workers)
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
        result = AdaptiveArray(self.memory_threshold, self.dtype, self.num_io_workers)
        data = self.to_array()
        result.append(data)
        return result
    
    def clear(self):
        """Clear all data"""
        with self._lock:
            self._chunks = []
            if self._use_mmap and hasattr(self, '_mmap_path') and self._mmap_path:
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
            self._mmap_path = None
    
    def __del__(self):
        """Cleanup memory-mapped files and thread pool"""
        self.clear()
        try:
            self._io_executor.shutdown(wait=False)
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
        result = AdaptiveArray(self.memory_threshold, dtype, self.num_io_workers)
        result.append(data)
        return result
    
    def flatten(self) -> 'AdaptiveArray':
        """Flatten array to 1D"""
        data = self.to_array().flatten()
        result = AdaptiveArray(self.memory_threshold, self.dtype, self.num_io_workers)
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
        result = AdaptiveArray(self.memory_threshold, self.dtype, self.num_io_workers)
        result.append(data)
        return result
    
    def where(self, condition: np.ndarray, x: Any, y: Any) -> 'AdaptiveArray':
        """Return elements chosen from x or y depending on condition"""
        data = np.where(condition, x, y)
        result = AdaptiveArray(self.memory_threshold, self.dtype, self.num_io_workers)
        result.append(data)
        return result
    
    def concatenate(self, others: List['AdaptiveArray'], axis: int = 0) -> 'AdaptiveArray':
        """Concatenate multiple AdaptiveArrays"""
        all_data = [self.to_array()]
        for other in others:
            all_data.append(other.to_array())
        
        # Parallel concatenation for large arrays
        total_size = sum(arr.shape[0] for arr in all_data)
        if total_size > 1000000:
            concatenated = self._parallel_concat(all_data)
        else:
            concatenated = np.concatenate(all_data, axis=axis)
        
        result = AdaptiveArray(self.memory_threshold, self.dtype, self.num_io_workers)
        result.append(concatenated)
        return result
    
    def split(self, indices_or_sections, axis: int = 0) -> List['AdaptiveArray']:
        """Split array into multiple sub-arrays"""
        data = self.to_array()
        splits = np.split(data, indices_or_sections, axis=axis)
        
        results = []
        for split_data in splits:
            result = AdaptiveArray(self.memory_threshold, self.dtype, self.num_io_workers)
            result.append(split_data)
            results.append(result)
        
        return results
    
    def pad(self, pad_width, mode: str = 'constant', **kwargs) -> 'AdaptiveArray':
        """Pad the array"""
        data = self.to_array()
        padded = np.pad(data, pad_width, mode=mode, **kwargs)
        
        result = AdaptiveArray(self.memory_threshold, self.dtype, self.num_io_workers)
        result.append(padded)
        return result
    
    def clip(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> 'AdaptiveArray':
        """Clip values to a range"""
        data = np.clip(self.to_array(), min_val, max_val)
        result = AdaptiveArray(self.memory_threshold, self.dtype, self.num_io_workers)
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
    
    def flush(self):
        """Flush any pending I/O operations"""
        if self._use_mmap and self._mmap is not None:
            self._mmap.flush()
    
    def close(self):
        """Close and cleanup resources"""
        self.flush()
        self.clear()
        self._io_executor.shutdown(wait=True)
