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
from numba import prange
from dask import delayed, compute
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.spatial.transform import RotationSpline, Rotation
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from pbrAudioRay.core.entity_manager import EntityManager
from pbrAudioRay.lib.band_interpolator_set import BandInterpolatorSet

@dataclass
class TimeVaryingIRGenerator:
    """
    Generates time-varying impulse responses from interpolators.
    Uses overlap-add convolution for smooth transitions.
    """
    entity_manager: EntityManager
    interpolators: Dict[Tuple[int, int], Dict[str, Any]]
    
    def __post_init__(self):
        self.config = self.entity_manager.get('config')
        self.sample_rate = self.config.system.sample_rate
        self.n_bands = len(self.entity_manager.get('frequency_bands').get_bands())
        
    @delayed
    def generate_ir_sequence(self,
                            source_idx: int,
                            output_idx: int,
                            duration: float) -> np.ndarray:
        """
        Generate a time-varying impulse response sequence.
        
        Args:
            source_idx: Source index
            output_idx: Output index
            duration: Duration in seconds
            
        Returns:
            Complex IR array (n_samples, n_bands)
        """
        key = (source_idx, output_idx)
        if key not in self.interpolators:
            raise ValueError(f"No interpolators for source {source_idx} -> output {output_idx}")
        
        interpolator_data = self.interpolators[key]
        band_sets = interpolator_data['band_interpolators']
        
        n_samples = int(duration * self.sample_rate)
        
        # Generate IR using SIMD
        ir = self._generate_ir_simd(band_sets, n_samples, self.sample_rate)
        
        return ir
    
    @staticmethod
    @nb.jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _generate_ir_simd(band_sets: List[BandInterpolatorSet],
                         n_samples: int,
                         sample_rate: int) -> np.ndarray:
        """
        SIMD-optimized IR generation from interpolators.
        
        Args:
            band_sets: List of BandInterpolatorSet objects
            n_samples: Number of output samples
            sample_rate: Audio sample rate
            
        Returns:
            Complex IR array (n_samples, n_bands)
        """
        n_bands = len(band_sets)
        ir = np.zeros((n_samples, n_bands), dtype=np.complex128)
        
        for sample in nb.prange(n_samples):
            time = sample / sample_rate
            
            for b in range(n_bands):
                band = band_sets[b]
                
                # Get interpolated values
                energy = 0.0
                phase = 0.0
                delay = 0.0
                
                if band.energy_interpolator is not None:
                    energy = band.energy_interpolator(time)
                    energy = max(energy, 0.0)
                
                if band.phase_interpolpolator is not None:
                    phase = band.phase_interpolator(time)
                    phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
                
                if band.delay_interpolator is not None:
                    delay = band.delay_interpolator(time)
                    delay = max(delay, 0.0)
                
                # Calculate sample index with fractional delay
                delay_samples = delay * sample_rate
                delay_int = int(delay_samples)
                delay_frac = delay_samples - delay_int
                
                if delay_int < n_samples:
                    # Linear interpolation for fractional delay
                    value = energy * np.exp(1j * phase)
                    
                    if delay_int + 1 < n_samples:
                        ir[delay_int, b] += value * (1.0 - delay_frac)
                        ir[delay_int + 1, b] += value * delay__frac
                    else:
                        ir[delay_int, b] += value
        
        return ir
    
    def save_interpolators(self, filepath: str):
        """
        Save interpolators to file for later use.
        
        Args:
            filepath: Path to save interpolators
        """
        import pickle
        
        # Convert interpolators to serializable format
        serializable_data = {}
        
        for key, data in self.interpolators.items():
            serializable_data[key] = {
                'band_interpolators': [],
                'rotation_spline': None
            }
            
            for band_set in data['band_interpolators']:
                # Save interpolator coefficients
                band_dict = {
                    'band_idx': band_set.band_idx,
                    'frame_times': band_set.frame_times.tolist()
                }
                
                if band_set.energy_interpolator is not None:
                    band_dict['energy_x'] = band_set.energy_interpolator.x.tolist()
                    band_dict['energy_y'] = band_set.energy_interpolator.y.tolist()
                
                if band_set.phase_interpolator is not None:
                    band_dict['phase_x'] = band_set.phase_interpolator.x.tolist()
                    band_dict['phase_y'] = band_set.phase_interpolator.y.tolist()
                
                if band_set.azimuth_interpolator is not None:
                    band_dict['azimuth_x'] = band_set.azimuth_interpolator.x.tolist()
                    band_dict['azimuth_y'] = band_set.azimuth_interpolator.y.tolist()
                
                if band_set.elevation_interpolator is not None:
                    band_dict['elevation_x'] = band_set.elevation_interpolator.x.tolist()
                    band_dict['elevation_y'] = band_set.elevation_interpolator.y.tolist()
                
                if band_set.delay_interpolator is not None:
                    band_dict['delay_x'] = band_set.delay_interpolator.x.tolist()
                    band_dict['delay_y'] = band_set.delay_interpolator.y.tolist()
                
                serializable_data[key]['band_interpolators'].append(band_dict)
            
            # Save rotation spline
            if data['rotation_spline'] is not None:
                rs = data['rotation_spline']
                serializable_data[key]['rotation_spline'] = {
                    'times': rs.times.tolist(),
                    'rotations': [r.as_quat().tolist() for r in rs.rotations]
                }
        
        with open(filepath, 'wb') as f:
            pickle.dump(serializable_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Saved interpolators to {filepath}")
    
    def load_interpolators(self, filepath: str):
        """
        Load interpolators from file.
        
        Args:
            filepath: Path to load interpolators from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            serializable_data = pickle.load(f)
        
        # Reconstruct interpolators
        for key, data in serializable_data.items():
            band_sets = []
            
            for band_dict in data['band_interpolators']:
                band_set = BandInterpolatorSet(
                    band_idx=band_dict['band_idx'],
                    frame_times=np.array(band_dict['frame_times'])
                )
                
                if 'energy_x' in band_dict:
                    band_set.energy_interpolator = CubicSpline(
                        np.array(band_dict['energy_x']),
                        np.array(band_dict['energy_y']),
                        bc_type='natural',
                        extrapolate=True
                    )
                
                if 'phase_x' in band_dict:
                    band_set.phase_interpolator = CubicSpline(
                        np.array(band_dict['phase_x']),
                        np.array(band_dict['phase_y']),
                        bc_type='natural',
                        extrapolate=True
                    )
                
                if 'azimuth_x' in band_dict:
                    band_set.azimuth_interpolator = PchipInterpolator(
                        np.array(band_dict['azimuth_x']),
                        np.array(band_dict['azimuth_y'])
                    )
                
                if 'elevation_x' in band_dict:
                    band_set.elevation_interpolator = PchipInterpolator(
                        np.array(band_dict['elevation_x']),
                        np.array(band_dict['elevation_y'])
                    )
                
                if 'delay_x' in band_dict:
                    band_set.delay_interpolator = CubicSpline(
                        np.array(band_dict['delay_x']),
                        np.array(band_dict['delay_y']),
                        bc_type='natural',
                        extrapolate=True
                    )
                
                band_sets.append(band_set)
            
            rotation_spline = None
            if data['rotation_spline'] is not None:
                rs_data = data['rotation_spline']
                times = np.array(rs_data['times'])
                rotations = Rotation.from_quat(np.array(rs_data['rotations']))
                rotation_spline = RotationSpline(times, rotations)
            
            self.interpolators[key] = {
                'band_interpolators': band_sets,
                'rotation_spline': rotation_spline
            }
        
        print(f"Loaded interpolators from {filepath}")
