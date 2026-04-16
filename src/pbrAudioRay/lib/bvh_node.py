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

class BVHNode:
    """Bounding Volume Hierarchy node for spatial acceleration"""
    def __init__(self, bbox_min, bbox_max, triangles=None, children=None):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.triangles = triangles if triangles is not None else []
        self.children = children if children is not None else []

@nb.njit
def build_bvh(triangles: np.ndarray, max_triangles_per_leaf: int = 32):
    """Build BVH for faster ray intersection"""
    # Simplified BVH construction - in practice use SAH
    n_triangles = triangles.shape[0]
    
    if n_triangles <= max_triangles_per_leaf:
        # Create leaf node
        bbox_min = np.min(triangles.reshape(-1, 3), axis=0)
        bbox_max = np.max(triangles.reshape(-1, 3), axis=0)
        return BVHNode(bbox_min, bbox_max, triangles)
    
    # Split along longest axis
    bbox_min = np.min(triangles.reshape(-1, 3), axis=0)
    bbox_max = np.max(triangles.reshape(-1, 3), axis=0)
    axis = np.argmax(bbox_max - bbox_min)
    
    # Sort triangles along axis
    centroids = np.mean(triangles, axis=1)
    sorted_indices = np.argsort(centroids[:, axis])
    
    # Split in middle
    mid = n_triangles // 2
    left_tris = triangles[sorted_indices[:mid]]
    right_tris = triangles[sorted_indices[mid:]]
    
    left_child = build_bvh(left_tris, max_trianglesangles_per_leaf)
    right_child = build_bvh(right_tris, max_triangles_per_leaf)
    
    return BVHNode(bbox_min, bbox_max, children=[left_child, right_child])
