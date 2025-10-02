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
import numba
from typing import List, Tuple, Optional, Dict
import soundfile as sf
from .soxel import SoxelGrid
from ..utils.math_utils import laplacian_3d

class WaveSolver:
    """
    Finite Difference Time Domain (FDTD) acoustic wave solver.
    """
    
    def __init__(self, 
                 dimensions: Tuple[int, int, int],
                 voxel_size: float,
                 sound_speed: float = 343.0,  # m/s
                 density: float = 1.2,       # kg/m³
                 cfl_number: float = 0.3):
        
        self.dimensions = dimensions
        self.voxel_size = voxel_size
        self.sound_speed = sound_speed
        self.density = density
        self.cfl_number = cfl_number
        
        # Calculate stable time step
        self.dt = self._calculate_stable_dt()
        
        # Initialize soxel grid
        self.soxel_grid = SoxelGrid(dimensions, voxel_size)
        
        # Simulation state
        self.pressure = np.zeros(dimensions, dtype=np.float32)
        self.velocity_x = np.zeros(dimensions, dtype=np.float32)
        self.velocity_y = np.zeros(dimensions, dtype=np.float32)
        self.velocity_z = np.zeros(dimensions, dtype=np.float32)
        
        # Material properties
        self.impedance = np.ones(dimensions, dtype=np.float32) * (sound_speed * density)
        self.absorption = np.zeros(dimensions, dtype=np.float32)
        
        # Sound sources
        self.sound_sources = []
        self.current_time = 0.0
        
    def _calculate_stable_dt(self) -> float:
        """Calculate stable time step using CFL condition."""
        dx = self.voxel_size
        dt = self.cfl_number * dx / (self.sound_speed * np.sqrt(3))
        return dt
    
    def add_sound_source(self, position: Tuple[int, int, int], 
                        wav_file: str, 
                        amplitude: float = 1.0):
        """Add a canned sound source from a mono WAV file."""
        # Load mono WAV file
        audio_data, sample_rate = sf.read(wav_file)
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]  # Take first channel if stereo
            
        self.sound_sources.append({
            'position': position,
            'audio_data': audio_data,
            'sample_rate': sample_rate,
            'amplitude': amplitude,
            'current_sample': 0
        })
    
    def set_impedance_field(self, impedance: np.ndarray):
        """Set the acoustic impedance field."""
        if impedance.shape != self.dimensions:
            raise ValueError("Impedance field must match grid dimensions")
        self.impedance = impedance.astype(np.float32)
    
    def set_absorption_field(self, absorption: np.ndarray):
        """Set the absorption coefficient field."""
        if absorption.shape != self.dimensions:
            raise ValueError("Absorption field must match grid dimensions")
        self.absorption = absorption.astype(np.float32)
    
    @numba.jit(nopython=True)
    def _update_velocity(self, pressure: np.ndarray, 
                        vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                        dt: float, dx: float, density: float):
        """Update velocity field using pressure gradient."""
        nx, ny, nz = pressure.shape
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    # Central differences for pressure gradient
                    dp_dx = (pressure[i+1, j, k] - pressure[i-1, j, k]) / (2 * dx)
                    dp_dy = (pressure[i, j+1, k] - pressure[i, j-1, k]) / (2 * dx)
                    dp_dz = (pressure[i, j, k+1] - pressure[i, j, k-1]) / (2 * dx)
                    
                    # Update velocity components
                    vx[i, j, k] -= (dt / density) * dp_dx
                    vy[i, j, k] -= (dt / density) * dp_dy
                    vz[i, j, k] -= (dt / density) * dp_dz
    
    @numba.jit(nopython=True)
    def _update_pressure(self, pressure: np.ndarray,
                        vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                        dt: float, dx: float, impedance: np.ndarray,
                        absorption: np.ndarray):
        """Update pressure field using velocity divergence."""
        nx, ny, nz = pressure.shape
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    # Velocity divergence
                    dvx_dx = (vx[i+1, j, k] - vx[i-1, j, k]) / (2 * dx)
                    dvy_dy = (vy[i, j+1, k] - vy[i, j-1, k]) / (2 * dx)
                    dvz_dz = (vz[i, j, k+1] - vz[i, j, k-1]) / (2 * dx)
                    
                    divergence = dvx_dx + dvy_dy + dvz_dz
                    
                    # Update pressure with absorption
                    c = impedance[i, j, k] / 1.2  # Convert impedance to sound speed
                    alpha = absorption[i, j, k]
                    
                    pressure[i, j, k] -= (dt * c**2 * 1.2 * divergence + 
                                         alpha * pressure[i, j, k] * dt)
    
    def step(self):
        """Perform one simulation step."""
        # Apply sound sources
        self._apply_sound_sources()
        
        # Update velocity field
        self._update_velocity(self.pressure, self.velocity_x, self.velocity_y, self.velocity_z,
                            self.dt, self.voxel_size, self.density)
        
        # Update pressure field
        self._update_pressure(self.pressure, self.velocity_x, self.velocity_y, self.velocity_z,
                            self.dt, self.voxel_size, self.impedance, self.absorption)
        
        # Apply boundary conditions (simple absorbing boundaries)
        self._apply_boundary_conditions()
        
        self.current_time += self.dt
    
    def _apply_sound_sources(self):
        """Apply sound source contributions to the pressure field."""
        for source in self.sound_sources:
            i, j, k = source['position']
            
            # Get current sample (with linear interpolation for sub-sample rates)
            sample_rate = source['sample_rate']
            sim_sample_rate = 1.0 / self.dt
            
            if sim_sample_rate > sample_rate:
                # Upsample needed - simple linear interpolation
                sample_idx = source['current_sample']
                if sample_idx < len(source['audio_data']) - 1:
                    t_frac = (self.current_time * sample_rate) % 1.0
                    sample_value = (1 - t_frac) * source['audio_data'][int(sample_idx)] + \
                                  t_frac * source['audio_data'][int(sample_idx) + 1]
                else:
                    sample_value = 0.0
            else:
                # Downsample - just take nearest sample
                sample_idx = int(self.current_time * sample_rate)
                if sample_idx < len(source['audio_data']):
                    sample_value = source['audio_data'][sample_idx]
                else:
                    sample_value = 0.0
            
            # Apply to pressure field
            self.pressure[i, j, k] += source['amplitude'] * sample_value
            
            source['current_sample'] = self.current_time * sample_rate
    
    def _apply_boundary_conditions(self):
        """Apply simple absorbing boundary conditions."""
        # Simple first-order absorbing boundaries
        boundary_strength = 0.1
        
        # X boundaries
        self.pressure[0, :, :] *= (1 - boundary_strength)
        self.pressure[-1, :, :] *= (1 - boundary_strength)
        
        # Y boundaries
        self.pressure[:, 0, :] *= (1 - boundary_strength)
        self.pressure[:, -1, :] *= (1 - boundary_strength)
        
        # Z boundaries
        self.pressure[:, :, 0] *= (1 - boundary_strength)
        self.pressure[:, :, -1] *= (1 - boundary_strength)
    
    def simulate(self, num_steps: int, save_interval: int = 1):
        """Run simulation for specified number of steps."""
        for step in range(num_steps):
            self.step()
            
            if step % save_interval == 0:
                # Save current state to soxel grid
                velocity = (self.velocity_x.copy(), self.velocity_y.copy(), self.velocity_z.copy())
                self.soxel_grid.add_frame(self.pressure.copy(), velocity)
                
                print(f"Step {step}/{num_steps}, Time: {self.current_time:.4f}s")
