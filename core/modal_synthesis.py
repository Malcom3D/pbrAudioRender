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
from scipy import linalg
from typing import List, Tuple

class ModalSynthesizer:
    """Modal synthesis for rigid body impacts"""
    
    def __init__(self, material_properties: dict[str, float]):
        self.material_properties = material_properties
    
    def compute_modes_3d(self, geometry: np.ndarray, youngs_modulus: float,
                        poisson_ratio: float, density: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute vibration modes for 3D geometry"""
        # Simplified modal analysis - in practice you'd use FEM
        n_nodes = geometry.shape[0]
        
        # Stiffness matrix (simplified)
        K = np.random.randn(n_nodes, n_nodes)  # Replace with actual FEM stiffness
        K = K.T @ K  # Make symmetric positive definite
        
        # Mass matrix (lumped)
        M = np.eye(n_nodes) * density
        
        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = linalg.eigh(K, M)
        
        # Convert to natural frequencies
        natural_frequencies = np.sqrt(eigenvalues) / (2 * np.pi)
        
        return natural_frequencies, eigenvectors
    
    def synthesize_impact(self, modes: np.ndarray, frequencies: np.ndarray,
                         impact_force: np.ndarray, damping_ratios: np.ndarray,
                         duration: float, sample_rate: int) -> np.ndarray:
        """Synthesize impact sound using modal superposition"""
        t = np.linspace(0, duration, int(duration * sample_rate))
        output = np.zeros_like(t)
        
        for i, (mode, freq, damping) in enumerate(zip(modes.T, frequencies, damping_ratios)):
            # Modal contribution
            amplitude = np.dot(mode, impact_force)
            decay = np.exp(-2 * np.pi * freq * damping * t)
            oscillation = np.sin(2 * np.pi * freq * t)
            
            output += amplitude * decay * oscillation
        
        return output
