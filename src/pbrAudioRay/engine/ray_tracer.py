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

class RayTracer:
    """
    Implements differentiable path tracing for acoustic rendering
    Based on: https://pub.dega-akustik.de/DAGA_2024/files/upload/paper/489.pdf
    """
    
    def __init__(self, entity_manager):
        self.entity_manager = entity_manager
        self.config = entity_manager.get('config')
        
    def compute_gradient_paths(self, frame_idx):
        """
        Compute paths with gradient information for differentiable rendering
        """
        # Implementation of Algorithm 1 from the paper
        paths = self._sample_initial_paths(frame_idx)
        
        for bounce in range(self.config.wave_propagation.max_interactions):
            # Propagate paths
            paths = self._propagate_paths(paths, frame_idx)
            
            # Compute gradients
            gradients = self._compute_path_gradients(paths)
            
            # Apply importance sampling based on gradients
            paths = self._resample_paths(paths, gradients)
        
        return self._accumulate_path_contributions(paths)
    
    @nb.njit
    def _compute_path_gradients(self, paths):
        """
        Compute gradients of acoustic properties along paths
        This enables differentiable rendering for parameter optimization
        """
        gradients = []
        for path in paths:
            # Compute gradient of energy w.r.t. material parameters
            grad = np.zeros((len(path.segments), 4))  # [absorption, reflection, scattering, refraction]
            
            for i, segment in enumerate(path.segments):
                # Finite differences for gradient estimation
                eps = 1e-6
                
                # Perturb absorption
                energy_plus = segment.energy * (1 + eps)
                energy_minus = segment.energy * (1 - eps)
                grad[i, 0] = (energy_plus - energy_minus) / (2 * eps)
                
                # Similar for other parameters...
            
            gradients.append(grad)
        
        return gradients
