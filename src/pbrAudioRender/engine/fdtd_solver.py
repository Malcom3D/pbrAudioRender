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
        for bands_idx in range(len(bands)-1):
            self.low = bands[bands_idx][0]
            self.high = bands[bands_idx][1]

        # Initialize interfaces and resonances
#        self.interface = Interface(config, self.gpu)
#        self.resonance = Resonance(config, self.gpu)

    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def _update_pressure(self, layer_pressure: np.ndarray, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, sound_speed: np.ndarray, density: np.ndarray) -> np.ndarray:
        """Update pressure field using velocity divergence"""
        new_pressure = np.zeros_like(pressure)

        voxel_volume = self.dx**3

        for i in nb.prange(1, pressure.shape[0]-1):
            for j in range(1, pressure.shape[1]-1):
                for k in range(1, pressure.shape[2]-1):
                    # Calculate velocity divergence
                    div_v = (
                        (vx[i+1, j, k] - vx[i-1, j, k]) / (2 * self.dx) +
                        (vy[i, j+1, k] - vy[i, j-1, k]) / (2 * self.dx) +
                        (vz[i, j, k+1] - vz[i, j, k-1]) / (2 * self.dx)
                    )

                    # Update pressure
                    c = sound_speed[i, j, k]
                    rho = density[i, j, k]
                    voxel_pressure = (layer_pressure[i, j, k]*voxel_volume + pressure[i, j, k]*voxel_volume)/(2*voxel_volume)
                    new_pressure[i, j, k] = (
                        voxel_pressure -
                        rho * c**2 * self.dt * div_v
                    )

        return new_pressure

    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def _update_velocity(self, layer_vx: np.ndarray, layer_vy: np.ndarray, layer_vz: np.ndarray, pressure: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, density: np.ndarray):
        """Update velocity fields using pressure gradient"""
        new_vx = np.zeros_like(vx)
        new_vy = np.zeros_like(vy)
        new_vz = np.zeros_like(vz)

        voxel_volume = self.dx**3

        for i in nb.prange(1, pressure.shape[0]-1):
            for j in range(1, pressure.shape[1]-1):
                for k in range(1, pressure.shape[2]-1):
                    rho = density[i, j, k]

                    # Update x-velocity
                    dp_dx = (pressure[i+1, j, k] - pressure[i-1, j, k]) / (2 * self.dx)
                    voxel_vx = (layer_vx[i, j, k]*voxel_volume + vx[i, j, k]*voxel_volume)/(2*voxel_volume)
                    new_vx[i, j, k] = vx - (self.dt / rho) * dp_dx

                    # Update y-velocity
                    dp_dy = (pressure[i, j+1, k] - pressure[i, j-1, k]) / (2 * self.dx)
                    voxel_vy = (layer_vy[i, j, k]*voxel_volume + vy[i, j, k]*voxel_volume)/(2*voxel_volume)
                    new_vy[i, j, k] = vy - (self.dt / rho) * dp_dy

                    # Update z-velocity
                    dp_dz = (pressure[i, j, k+1] - pressure[i, j, k-1]) / (2 * self.dx)
                    voxel_vz = (layer_vz[i, j, k]*voxel_volume + vz[i, j, k]*voxel_volume)/(2*voxel_volume)
                    new_vz[i, j, k] = vz - (self.dt / rho) * dp_dz

        return new_vx, new_vy, new_vz

    def _update_layer(self, layer: AcousticLayer, new_pressure: np.ndarray, new_vx: np.ndarray, new_vy: np.ndarray, new_vz: np.ndarray):
        for i in range(layer.shape[0]):
            for j in range(layer.shape[1]):
                for k in range(layer.shape[2]):
                    layer[i,j,k] = FrequencyLimitedField(self.low_freq, self.high_freq, new_pressure[i,j,k], VelocityVectors(new_vx[i,j,k],new_vy[i,j,k],new_vz[i,j,k]))

    def update_step(self):
        """Perform one FDTD update step"""
        soxel_grid = self.entity_manager.get('soxel_grid')
        # Get acoustic pressure and velocity vectors for this frequency band
        shm_pressure = soxel_grid.get_shm_array('pressure', self.low_freq, self.high_freq)
        shm_vx = soxel_grid.get_shm_array('vx', self.low_freq, self.high_freq)
        shm_vy = soxel_grid.get_shm_array('vy', self.low_freq, self.high_freq)
        shm_vz = soxel_grid.get_shm_array('vz', self.low_freq, self.high_freq)

        # Get acoustic properties for this frequency band
        shm_sound_speed = soxel_grid.get_shm_array('sound_speed')
        shm_density = soxel_grid.get_shm_array('density')

        # Get owned layer
        for index in self.layer_manger.layers.keys():
            if str(self.low_freq) in self.layer_manger.layers[index].name and str(self.high_freq) in self.layer_manger.layers[index].name:
                layer = self.layer_manger.layers[index]
                shm_layer_pressure = get_shm_layer(index, 'pressure')
                shm_layer_vx = get_shm_layer(index, 'vx')
                shm_layer_vy = get_shm_layer(index, 'vy')
                shm_layer_vz = get_shm_layer(index, 'vz')

        # Update pressure and velocity vectors
        new_pressure = self._update_pressure(shm_layer_pressure, shm_pressure, shm_vx, shm_vy, shm_vz, shm_sound_speed, shm_density)
        new_vx, new_vy, new_vz = self._update_velocity(shm_layer_vx, shm_layer_vy, shm_layer_vz, new_pressure, shm_vx, shm_vy, shm_vz, shm_density)

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

    
    def _run_fdtd_solver(self):
        frequency_bands = self.entity_manager.get('frequency_bands')
        self.bands = frequency_bands.get_bands()
        for bands_idx in range(len(bands)-1):
            # Layer selection or init
            wave_propagators = self.entity_manager.get('wave_propagators', self.idx)
            layer_manger = wave_propagators.layer_manger
            if band_idx > len(layer_manger.layers):
                layer_manger.add_new('fdtd', bands_idx)
            self.layer = layer_manger.get_layer('fdtd', bands_idx)
            # Create FDTD solvers for all bands
            self.fdtd_solvers.append(FDTDSolver(self.entity_manager, self.idx, bands_idx))
            # Initialize interface and resonance for this frequency band
#            self.interface = Interface(self.entity_manager, self.idx)
#            self.resonance = Resonance(self.entity_manager, self.idx)


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
        self._run_fdtd_solver()
        updated_layer = self.fdtd_solver.update_step(layer_manager, soxel_grid, source_audio)
        
        # Step 2: Interface interactions
        updated_layer = self.interface.update_step(updated_layer, soxel_grid)
        
        
        # Step 3: Resonance effects
        updated_layer = self.resonance.update_step(updated_layer, soxel_grid)
        
        return updated_layer
