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

from ..core.entity_manager import EntityManager
from ..lib.output_data import OutputData
from ..lib.band_interpolator_set import BandInterpolatorSet
from ..lib.functions import _cartesian_to_spherical

@dataclass
class InterpolationEngine:
    """
    Engine for creating smooth interpolators from per-frame output data.
    Uses SIMD, numba, and dask for performance.
    """
    entity_manager: EntityManager
    
    def __post_init__(self):
        self.config = self.entity_manager.get('config')
        self.frequency_bands = self.entity_manager.get('frequency_bands')
        self.n_bands = len(self.frequency_bands.get__bands())
        self.fps = self.config.system.fps
        self.sample_rate = self.config.system.sample_rate
        
        # Pre-allocate storage for interpolators
        self.band_interpolators: Dict[Tuple[int, int], List[BandInterpolatorSet]] = {}
        # Key: (source_idx, output_idx), Value: List[BandInterpolatorSet] per band
        
    @delayed
    def compute_interpolators_for_pair(self, 
                                      source_idx: int, 
                                      output_idx: int,
                                      output_datas: List[OutputData]) -> List[BandInterpolatorSet]:
        """
        Compute interpolators for a source-output pair across all bands.
        
        Args:
            source_idx: Source index
            output_idx: Output index
            output_datas: List of OutputData objects (one per frame)
            
        Returns:
            List of BandInterpolatorSet (one per frequency band)
        """
        if len(output_datas) == 0:
            return []
        
        # Sort by frame index
        output_datas.sort(key=lambda x: x.frame_idx if hasattr(x, 'frame_idx') else 0)
        
        # Extract frame times
        frame_times = np.array([od.frame_idx / self.fps for od in output_datas], dtype=np.float64)
        
        band_sets = []
        
        # Process each band independently using SIMD
        for band_idx in range(self.n_bands):
            band_set = self._compute_band_interpolator(
                band_idx, frame_times, output_datas
            )
            band_sets.append(band_set)
            
        return band_sets
    
    def _compute_band_interpolator(self,
                                   band_idx: int,
                                   frame_times: np.ndarray,
                                   output_datas: List[OutputData]) -> BandInterpolatorSet:
        """
        Compute interpolators for a single frequency band.
        
        Args:
            band_idx: Frequency band index
            frame_times: Array of frame times
            output_datas: List of OutputData objects
            
        Returns:
            BandInterpolatorSet for this band
        """
        n_frames = len(output_datas)
        
        # Pre-allocate arrays for SIMD processing
        energies = np.zeros(n_frames, dtype=np.float64)
        phases = np.zeros(n_frames, dtype=np.float64)
        azimuths = np.zeros(n_frames, dtype=np.float64)
        elevations = np.zeros(n_frames, dtype=np.float64)
        delays = np.zeros(n_frames, dtype=np.float64)
        
        # Extract data using SIMD-optimized function
        self._extract_band_data_simd(
            output_datas, band_idx, 
            energies, phases, azimuths, elevations, delays
        )
        
        # Create interpolators
        band_set = BandInterpolatorSet(band_idx=band_idx, frame_times=frame_times)
        
        try:
            # Energy interpolator (always positive, use PCHIP to avoid overshoot)
            if np.all(energies >= 0) and not np.all(energies == energies[0]):
                band_set.energy_interpolator = PchipInterpolator(frame_times, energies)
            else:
                band_set.energy_interpolator = CubicSpline(frame_times, energies, 
                                                          bc_type='natural',
                                                          extrapolate=True)
            
            # Phase interpolator (wrap to [-π, π])
            if not np.all(phases == phases[0]):
                # Unwrap phases for smooth interpolation
                unwrapped_phases = np.unwrap(phases)
                band_set.phase_interpolator = CubicSpline(frame_times, unwrapped_phases,
                                                         bc_type='natural',
                                                         extrapolate=True)
            
            # Azimuth interpolator (handle circular nature)
            if not np.all(azimuths == azimuths[0]):
                # Use PCHIP for azimuth to avoid overshoot
                band_set.azimuth_interpolator = PchipInterpolator(frame_times, azimuths)
            
            # Elevation interpolator
            if not np.all(elevations == elevations[0]):
                band_set.elevation_interpolator = PchipInterpolator(frame_times, elevations)
            
            # Delay interpolator
            if not np.all(delays == delays[0]):
                band_set.delay_interpolator = CubicSpline(frame_times, delays,
                                                        bc_type='natural',
                                                        extrapolate=True)
                
        except Exception as e:
            print(f"Warning: Could not create interpolators for band {band_idx}: {e}")
        
        return band_set
    
    @staticmethod
    @nb.jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _extract_band_data_simd(output_datas: List[OutputData],
                                band_idx: int,
                                energies: np.ndarray,
                                phases: np.ndarray,
                                azimuths: np.ndarray,
                                elevations: np.ndarray,
                                delays: np.ndarray):
        """
        SIMD-optimized extraction of band data from OutputData objects.
        
        Args:
            output_datas: List of OutputData objects
            band_idx: Frequency band index
            energies: Output array for energies
            phases: Output array for phases
            azimuths: Output array for azimuths
            elevations: Output array for elevations
            delays: Output array for delays
        """
        n_frames = len(output_datas)
        
        for i in nb.prange(n_frames):
            od = output_datas[i]
            
            # Get energy and phase for this band
            if od.energies.shape[0] > 0:
                if od.energies.ndim == 2 and od.energies.shape[1] > band_idx:
                    energies[i] = np.mean(od.energies[:, band_idx])
                else:
                    energies[i] = np.mean(od.energies)
            
            if od.phases.shape[0] > 0:
                if od.phases.ndim == 2 and od.phases.shape[1] > band_idx:
                    phases[i] = np.mean(od.phases[:, band_idx])
                else:
                    phases[i] = np.mean(od.phases)
            
            # Get delay
            if od.delay.shape[0] > 0:
                delays[i] = np.mean(od.delay)
            
            # Convert directions to spherical coordinates
            if od.directions.shape[0] > 0:
                # Average direction
                avg_dir = np.mean(od.directions, axis=0)
                norm = np.sqrt(avg_dir[0]**2 + avg_dir[1]**2 + avg_dir[2]**2)
                
                if norm > 1e-10:
                    x, y, z = avg_dir[0] / norm, avg_dir[1] / norm, avg_dir[2] / norm
                    
                    # Convert to spherical coordinates
                    azimuths[i] = np.arctan2(y, x)
                    elevations[i] = np.arcsin(z)
    
    @delayed
    def compute_rotation_interpolator(self,
                                     source_idx: int,
                                     output_idx: int,
                                     output_datas: List[OutputData]) -> Optional[RotationSpline]:
        """
        Compute rotation interpolator for listener orientation.
        
        Args:
            source_idx: Source index
            output_idx: Output index
            output_datas: List of OutputData objects
            
        Returns:
            RotationSpline for smooth orientation interpolation
        """
        if len(output_datas) == 0:
            return None
        
        # Sort by frame index
        output_datas.sort(key=lambda x: x.frame_idx if hasattr(x, 'frame_idx') else 0)
        
        n_frames = len(output_datas)
        frame_times = np.array([od.frame_idx / self.fps for od in output_datas], dtype=np.float64)
        
        # Extract average directions per frame
        rotations = []
        valid_frames = []
        
        for i, od in enumerate(output_datas):
            if od.directionsctions.shape[0] > 0:
                avg_dir = np.mean(od.directions, axis=0)
                norm = np.linalg.norm(avg_dir)
                
                if norm > 1e-10:
                    # Create rotation from forward direction
                    forward = avg_dir / norm
                    
                    # Create rotation matrix (Z-up convention)
                    up = np.array([0.0, 0.0, 1.0])
                    right = np.cross(forward, up)
                    
                    if np.linalg.norm(right) > 1e-10:
                        right = right / np.linalg.norm(right)
                        up = np.cross(right, forward)
                        
                        rot_matrix = np.column_stack([right, up, forward])
                        rotations.append(Rotation.from_matrix(rot_matrix))
                        valid_frames.append(i)
        
        if len(rotations) < 2:
            return None
        
        # Create rotation spline
        valid_times = frame_times[valid_frames]
        rotations_array = Rotation.concatenate(rotations)
        
        try:
            return RotationSpline(valid_times, rotations_array)
        except Exception as e:
            print(f"Warning: Could not create rotation spline: {e}")
            return None
    
    def build_all_interpolators(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Build interpolators for all source-output pairs.
        
        Returns:
            Dictionary with interpolators for each source-output pair
        """
        # Get output data from entity manager
        output_datas = self.entity_manager.get('output_datas')
        
        if output_datas is None:
            print("No output data found")
            return {}
        
        # Group output data by source-output pair
        pair_data: Dict[Tuple[int, int], List[OutputData]] = {}
        
        for od_idx, od in output_datas.items():
            # We need to know the source and output indices
            # This assumes OutputData has source_idx and output_idx attributes
            source_idx = getattr(od, 'source_idx', None)
            output_idx = getattr(od, 'output_idx', None)
            
            if source_idx is not None and output_idx is not None:
                key = (source_idx, output_idx)
                if key not in pair_data:
                    pair_data[key] = []
                pair_data[key].append(od)
        
        # Compute interpolators for each pair in parallel
        tasks = []
        for key, data_list in pair_data.items():
            source_idx, output_idx = key
            
            # Compute band interpolators
            tasks.append(
                self.compute_interpolators_for_pair(source_idx, output_idx, data_list)
            )
            
            # Compute rotation interpolator
            tasks.append(
                self.compute_rotation_interpolator(source_idx, output_idx, data_list)
            )
        
        # Execute all tasks in parallel
        results = compute(*tasks)
        
        # Organize results
        all_interpolators = {}
        result_idx = 0
        
        for key in pair_data.keys():
            source_idx, output_idx = key
            
            band_sets = results[result_idx]
            result_idx += 1
            
            rotation_spline = results[result_idx]
            result_idx += 1
            
            all_interpolators[key] = {
                'band_interpolators': band_sets,
                'rotation_spline': rotation_spline
            }
        
        # Store in entity manager
        self.entity_manager.register('interpolators', all_interpolators)
        
        return all_interpolators
    
    @staticmethod
    @nb.jit(nopython=True, fastmath=True, cache=True)
    def interpolate_at_time_simd(band_sets: List[BandInterpolatorSet],
                                 time: float,
                                 n_bands: int) -> np.ndarray:
        """
        SIMD-optimized interpolation at a given time for all bands bands.
        
        Args:
            band_sets: List of BandInterpolatorSet objects
            time: Time in seconds
            n_bands: Number of frequency bands
            
        Returns:
            Array of interpolated complex values (n_bands,)
        """
        result = np.zeros(n_bands, dtype=np.complex128)
        
        for b in nb.prange(n_bands):
            band = band_sets[b]
            
            # Get interpolated values
            energy = 0.0
            phase = 0.0
            
            if band.energy_interpolator is not None:
                energy = band.energy_interpolator(time)
                energy = max(energy, 0.0)  # Ensure non-negative
            
            if band.phase_interpolator is not None:
                phase = band.phase_interpolator(time)
                # Wrap phase to [-π, π]
                phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
            
            # Create complex representation
            result[b] = energy * np.exp(1j * phase)
        
        return result
