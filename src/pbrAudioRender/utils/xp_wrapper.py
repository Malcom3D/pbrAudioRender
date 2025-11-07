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
import warnings
from functools import wraps
import time

class XpWrapper:
    """
    Advanced wrapper with performance monitoring and additional utilities.
    """
    
    def __init__(self, enable_numba=True):
        self._numba_available = False
        self._njit = None
        self._numpy_module = np
        self._performance_stats = {}
        
        if enable_numba:
            self._setup_numba()
        
    def _setup_numba(self):
        """Setup numba with enhanced error handling"""
        try:
            import numba
            from numba import njit, prange
            self._numba_available = True
            self._njit = njit
            self._prange = prange
            self._numba_module = numba
            self._performance_stats['numba_enabled'] = True
        except ImportError:
            self._numba_available = False
            self._performance_stats['numba_enabled'] = False
            warnings.warn(
                "Numba not available. Using pure numpy. "
                "Install with: pip install numba",
                ImportWarning
            )
    
    def jit(self, func=None, **kwargs):
        """
        Enhanced JIT decorator with performance tracking.
        """
        if func is None:
            return lambda f: self.jit(f, **kwargs)
        
        if self._numba_available:
            jitted_func = self._njit(**kwargs)(func)
            
            # Add performance tracking wrapper
            @wraps(func)
            def timed_wrapper(*args, **kwargs):
                start_time = time.time()
                result = jitted_func(*args, **kwargs)
                end_time = time.time()
                
                # Track performance
                func_name = func.__name__
                if func_name not in self._performance_stats:
                    self._performance_stats[func_name] = []
                self._performance_stats[func_name].append(end_time - start_time)
                
                return result
            return timed_wrapper
        else:
            # Fallback with timing
            @wraps(func)
            def timed_wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                func_name = func.__name__
                if func_name not in self._performance_stats:
                    self._performance_stats[func_name] = []
                self._performance_stats[func_name].append(end_time - start_time)
                
                return result
            return timed_wrapper
    
    def vectorize(self, signature=None, **kwargs):
        """
        Numba vectorize decorator with fallback.
        """
        if self._numba_available:
            try:
                from numba import vectorize
                return vectorize(signature, **kwargs)
            except ImportError:
                pass
        
        # Fallback: create a dummy vectorize decorator
        def dummy_vectorize(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return dummy_vectorize
    
    def get_performance_stats(self):
        """Get performance statistics for JIT functions"""
        return self._performance_stats.copy()
    
    def clear_performance_stats(self):
        """Clear performance statistics"""
        self._performance_stats.clear()
    
    def __getattr__(self, name):
        """Delegate to numpy, with special handling for common functions"""
        return getattr(self._numpy_module, name)
    
    @property
    def has_numba(self):
        return self._numba_available
    
    @property
    def prange(self):
        """Get prange for parallel loops (falls back to range)"""
        if self._numba_available:
            return self._prange
        else:
            return range

# Create advanced instance
#xp = XpWrapper()
