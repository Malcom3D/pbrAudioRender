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

import warnings

from dask import delayed, compute

import numpy as np
import numba as nb
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from core.entity_manager import EntityManager
from lib.acoustic_layer import AcousticLayer
from lib.acoustic_field import FrequencyLimitedField, VelocityVectors

@dataclass
class FDTDSolver:
    """Basic FDTD acoustic wave solver for single frequency band"""
    entity_manager: EntityManager
    idx: int
    bands_idx: int
    
    def __post_init__(self):
        config = self.entity_manager.get('config')

        # FDTD parameters
        self.dx = config.acoustic_domain.voxel_size
        self.dt = 1.0 / config.acoustic_domain.sample_rate

        # Get low and high frequency
        frequency_bands = self.entity_manager.get('frequency_bands')
        bands = frequency_bands.get_bands()
        self.low_freq = bands[self.bands_idx][0]
        self.high_freq = bands[self.bands_idx][1]

        # Initialize interfaces and resonances
        self.interface = Interface(config, self.gpu)
        self.resonance = Resonance(config, self.gpu)

    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def _update_pressure(layer_pressure: np.ndarray, soxel_pressure: np.ndarray, soxel_vx: np.ndarray, soxel_vy: np.ndarray, soxel_vz: np.ndarray, sound_speed: np.ndarray, density: np.ndarray, dt: float, dx: float) -> np.ndarray:
        """Update pressure field using velocity divergence"""
        new_pressure = np.zeros_like(soxel_pressure)

        voxel_volume = dx**3

        for i in nb.prange(1, soxel_pressure.shape[0]-1):
            for j in range(1, soxel_pressure.shape[1]-1):
                for k in range(1, soxel_pressure.shape[2]-1):
                    # Calculate velocity divergence
                    div_v = (
                        (soxel_vx[i+1, j, k] - soxel_vx[i-1, j, k]) / (2 * dx) +
                        (soxel_vy[i, j+1, k] - soxel_vy[i, j-1, k]) / (2 * dx) +
                        (soxel_vz[i, j, k+1] - soxel_vz[i, j, k-1]) / (2 * dx)
                    )

                    # Update pressure
                    c = sound_speed[i, j, k]
                    rho = density[i, j, k]
                    _pressure = (layer_pressure[i, j, k]*voxel_volume + soxel_pressure[i, j, k]*voxel_volume)/(2*voxel_volume)
                    new_pressure[i, j, k] = (
                        _pressure -
                        rho * c**2 * dt * div_v
                    )

        return new_pressure

    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def _update_velocity(layer_vx: np.ndarray, layer_vy: np.ndarray, layer_vz: np.ndarray, soxel_pressure: np.ndarray, soxel_vx: np.ndarray, soxel_vy: np.ndarray, soxel_vz: np.ndarray, density: np.ndarray, dt: float, dx: float):
        """Update velocity fields using pressure gradient"""
        new_vx = np.zeros_like(soxel_vx)
        new_vy = np.zeros_like(soxel_vy)
        new_vz = np.zeros_like(soxel_vz)

        voxel_volume = dx**3

        for i in nb.prange(1, soxel_pressure.shape[0]-1):
            for j in range(1, soxel_pressure.shape[1]-1):
                for k in range(1, soxel_pressure.shape[2]-1):
                    rho = density[i, j, k]

                    # Update x-velocity
                    dp_dx = (soxel_pressure[i+1, j, k] - soxel_pressure[i-1, j, k]) / (2 * dx)
                    _vx = (layer_vx[i, j, k]*voxel_volume + soxel_vx[i, j, k]*voxel_volume)/(2*voxel_volume)
                    new_vx[i, j, k] = _vx - (dt / rho) * dp_dx

                    # Update y-velocity
                    dp_dy = (soxel_pressure[i, j+1, k] - soxel_pressure[i, j-1, k]) / (2 * dx)
                    _vy = (layer_vy[i, j, k]*voxel_volume + soxel_vy[i, j, k]*voxel_volume)/(2*voxel_volume)
                    new_vy[i, j, k] = _vy - (dt / rho) * dp_dy

                    # Update z-velocity
                    dp_dz = (soxel_pressure[i, j, k+1] - soxel_pressure[i, j, k-1]) / (2 * dx)
                    _vz = (layer_vz[i, j, k]*voxel_volume + soxel_vz[i, j, k]*voxel_volume)/(2*voxel_volume)
                    new_vz[i, j, k] = _vz - (dt / rho) * dp_dz

        return new_vx, new_vy, new_vz

    def _update_layer(self, layer: AcousticLayer, new_pressure: np.ndarray, new_vx: np.ndarray, new_vy: np.ndarray, new_vz: np.ndarray):
        for i in range(layer.shape[0]):
            for j in range(layer.shape[1]):
                for k in range(layer.shape[2]):
                    velocity_vectors = VelocityVectors(new_vx[i,j,k],new_vy[i,j,k],new_vz[i,j,k])
                    layer.field[i,j,k] = FrequencyLimitedField(low_freq=self.low_freq, high_freq=self.high_freq, pressure=new_pressure[i,j,k], velocity=velocity_vectors)

    def update(self):
        """Perform one FDTD update step"""
        soxel_grid = self.entity_manager.get('soxel_grid')
        # Get acoustic pressure and velocity vectors for this frequency band
        soxel_pressure = soxel_grid.get_array('pressure', self.low_freq, self.high_freq)
        soxel_vx = soxel_grid.get_array('vx', self.low_freq, self.high_freq)
        soxel_vy = soxel_grid.get_array('vy', self.low_freq, self.high_freq)
        soxel_vz = soxel_grid.get_array('vz', self.low_freq, self.high_freq)

        # Get acoustic properties for this frequency band
        sound_speed = soxel_grid.get_array('sound_speed')
        density = soxel_grid.get_array('density')

        # Get owned layer
        wave_propagator = self.entity_manager.get('wave_propagators', self.idx)
        layer_manager = wave_propagator.layer_manager
        layer = layer_manager.get_layer('fdtd', self.bands_idx)
        layer_pressure = layer_manager.get_array('fdtd', self.bands_idx, 'pressure')
        layer_vx = layer_manager.get_array('fdtd', self.bands_idx, 'vx')
        layer_vy = layer_manager.get_array('fdtd', self.bands_idx, 'vy')
        layer_vz = layer_manager.get_array('fdtd', self.bands_idx, 'vz')

        # Update pressure and velocity vectors
        new_pressure = self._update_pressure(layer_pressure, soxel_pressure, soxel_vx, soxel_vy, soxel_vz, sound_speed, density, self.dt, self.dx)
        new_vx, new_vy, new_vz = self._update_velocity(layer_vx, layer_vy, layer_vz, new_pressure, soxel_vx, soxel_vy, soxel_vz, density, self.dt, self.dx)

        # Update owned layer
        self._update_layer(layer, new_pressure, new_vx, new_vy, new_vz)

@dataclass
class FDTDManager:
    """Multi-band FDTD solver managing multiple frequency-dependent solvers"""
    entity_manager: EntityManager
    idx: int
 
    def __post_init__(self):
        # FDTD parameters
        config = self.entity_manager.get('config')
        dx = config.acoustic_domain.voxel_size
        dt = 1/config.acoustic_domain.sample_rate
        max_sound_speed = config.fdtd.max_sound_speed # Conservative estimate
        courant_number = config.fdtd.courant_number
        # Stability check
        self._check_stability(max_sound_speed, courant_number, dt, dx)

    @delayed
    def _fdtd_solver_update(self, bands_idx):
        # Layer selection or init
        wave_propagator = self.entity_manager.get('wave_propagators', self.idx)
        layer_manger = wave_propagator.layer_manager
        layer_manger.add_new('fdtd', bands_idx)
        layer = layer_manger.get_layer('fdtd', bands_idx)

        # Create FDTD solvers for all bands
        fdtd_solver = FDTDSolver(self.entity_manager, self.idx, bands_idx)
        fdtd_solver.update()

        # Initialize interface and resonance for this frequency band
#        self.interface = Interface(self.entity_manager, self.idx)
#        self.resonance = Resonance(self.entity_manager, self.idx)

    def _check_stability(self, max_sound_speed: float, courant_number: float, dt: float, dx: float):
        """Check FDTD stability conditions"""
        courant_condition = max_sound_speed * dt / dx

        if courant_condition > courant_number:
            warnings.warn(
                f"FDTD stability warning: Courant number {courant_condition:.3f} "
                f"exceeds stability limit {courant_number:.3f}"
            )

    def update(self):
        """Perform multi-band update step with physical interactions"""
        # Step 1: Basic FDTD update
        frequency_bands = self.entity_manager.get('frequency_bands')
        bands = frequency_bands.get_bands()
        tasks = [self._fdtd_solver_update(bands_idx) for bands_idx in range(len(bands))]
        compute(*tasks)
        
        # Step 2: Interface interactions
#        updated_layer = self.interface.update(updated_layer, soxel_grid)
        
        
        # Step 3: Resonance effects
#        updated_layer = self.resonance.update(updated_layer, soxel_grid)
        
#        return updated_layer
