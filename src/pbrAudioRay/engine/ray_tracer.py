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
import trimesh
from dask import delayed, compute
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

from embreex import rtcore_scene as rtcs
from embreex.mesh_construction import TriangleMesh

from pbrAudioRay.core.entity_manager import EntityManager
from pbrAudioRay.engine.interface import InterfaceManager
from pbrAudioRay.lib.output_data import OutputData

@dataclass
class AcousticRayTracer:
    """Main class for acoustic ray tracing."""
    entity_manager: EntityManager
    geometry_data: Any
    material_properties: Any
    medium_properties: Any
    ray_data: Any
    recursion_idx: int = 0

    def __post_init__(self):
        """Initialize ray directions using Fibonacci sphere distribution."""
        config = self.entity_manager.get('config')
        n_rays = config.system.number_of_rays
        self.max_interactions = config.wave_propagation.max_interactions

        # Fibonacci sphere
        phi = np.pi * (3. - np.sqrt(5.))
        theta = phi * np.arange(n_rays)
        z = np.linspace(1/n_rays - 1, 1 - 1/n_rays, n_rays)
        radius = np.sqrt(1 - z * z)
        y = radius * np.sin(theta)
        x = radius * np.cos(theta)

        directions = np.array(list(zip(x, y, z)), dtype=np.float32)

        # Set initial directions towards destinations
        main_dir = self.ray_data.destinations[0] - self.ray_data.origins[0]
        main_dir_norm = np.linalg.norm(main_dir)
        main_dir_norm = max(main_dir_norm, 1e-10)
        main_dir = main_dir / main_dir_norm

        directions[0] = main_dir

        self.ray_data.directions = directions
        self.ray_data.energies = np.full((n_rays, 1), 1.0, dtype=np.float32)
        self.ray_data.phases = np.full((n_rays, 1), 0.0, dtype=np.float32)
        self.ray_data.delay = np.full((n_rays, 1), 0.0, dtype=np.float32)

        # Initialize Ray output
        self.output_data = OutputData(bands_idx=self.ray_data.bands_idx)

    @delayed
    def compute(self):
        # Initialize interface manager
        self.interface = InterfaceManager(self.entity_manager, self.geometry_data, self.material_properties, self.medium_properties, self.ray_data, self.output_data)

        # Initialize EmbreeX Scene
        self.scene = rtcs.EmbreeScene()

        # Create EmbreeX mesh scene
        embree_mesh = TriangleMesh(self.scene, self.geometry_data.mesh_info)

        # Start recursive ray tracing
        self._ray_tracing_loop()

        # Return output data for ir computations
        return self.output_data

    def _ray_tracing_loop(self):
        """Recursive ray tracing loop."""
        res = self.scene.run(self.ray_data.origins.astype(np.float32), self.ray_data.directions.astype(np.float32), output=1)

        ray_inter = res["geomID"] >= 0
        print(f"Recursion {self.recursion_idx}: {sum(ray_inter)} rays intersect geometry (over {self.ray_data.origins.shape[0]})")

        if not np.any(ray_inter) or self.recursion_idx == self.max_interactions:
            return self.output_data

        # Process intersections
        self.interface.compute(res, ray_inter)

        # Apply energy threshold
        self._apply_energy_threshold()

        # Recursive call
        self.recursion_idx += 1
        if self.ray_data.origins.shape[0] > 0:
            self._ray_tracing_loop()

    def _apply_energy_threshold(self):
        """Apply energy threshold to terminate low-energy rays."""
        termination_energy = 1e-16
        termination_mask = self.ray_data.energies > termination_energy

        self.ray_data.origins = self.ray_data.origins[termination_mask.reshape(-1,)]
        self.ray_data.directions = self.ray_data.directions[termination_mask.reshape(-1,)]
        self.ray_data.energies = self.ray_data.energies[termination_mask].reshape(-1, 1)
        self.ray_data.phases = self.ray_data.phases[termination_mask].reshape(-1, 1)
        self.ray_data.delay = self.ray_data.delay[termination_mask].reshape(-1, 1)

        n_terminated = np.count_nonzero(~termination_mask)
        if n_terminated > 0:
            print(f'Terminated {n_terminated} rays below energy threshold')

