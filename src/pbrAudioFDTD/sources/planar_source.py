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


import os
from pbrAudioCommon import np
from typing import Tuple, List
from dataclasses import dataclass, field

from ..core.entity_manager import EntityManager
from ..lib.functions import _audio_to_npz, _get_position, _get_rotation, _world_to_grid, _is_in_bounds, _cartesian_to_spherical
from ..lib.acoustic_field import VelocityVectors, AcousticField
from ..lib.soxel import Soxel

@dataclass
class PlanarSource:
    """
    Planar source with directional orientation
    """
    entity_manager: EntityManager
    idx: int

    def __post_init__(self):
        """
        Convert audio_file for source.idx to frequency dependent np.ndarray in npz file.
        """
        config = self.entity_manager.get('config')
        frequency_bands = self.entity_manager.get('frequency_bands')
        bands = frequency_bands.get_bands()
        self.frames = self.entity_manager.get('frames')
        self.shape = config.acoustic_domain.shape
        self.voxel_size = config.acoustic_domain.voxel_size
        self.grid_geometry = config.acoustic_domain.geometry
        self.sound_speed = config.acoustic_domain.acoustic_shader.sound_speed
        self.density = config.acoustic_domain.acoustic_shader.density
        grid_sample_rate = config.acoustic_domain.sample_rate

        for source_config in config.sources:
            if source_config.idx == self.idx:
                self.source_config = source_config

        audio_file = self.source_config.audio_file
        npz_path = os.path.join(config.system.cache_path, 'filtered_audio')
        audio_npz = str(self.source_config.idx) + '.npz'

        audio_npz = _audio_to_npz(npz_path, audio_file, audio_npz, grid_sample_rate, bands)

        # Open for get_field function
        fd_samples = np.load(audio_npz)
        fd_samples.allow_pickle = True
        self.fd_samples = fd_samples[fd_samples.files[0]]

    def get_soxels(self):
        """Voxelize a sound source into the grid with directional orientation"""

        current_frame = self.frames.get()

        center_pos = _get_position(self.source_config.position_file, current_frame)
        center = _world_to_grid(self.voxel_size, self.grid_geometry, center_pos)
        
        # Get rotation for directional orientation
        rotation = _get_rotation(self.source_config.rotation_file, current_frame)
        rotation_matrix = self._euler_to_rotation_matrix(rotation)
        
        # Rotate vertices to match orientation
        vertices = self.source_config.geometry
        rotated_vertices = self._rotate_vertices(vertices, rotation_matrix, center_pos)
        
        voxels = self._get_planar_voxels(rotated_vertices)

        # Update soxels at source positions
        soxels = []
        for voxel in voxels:
            i, j, k = voxel
            if _is_in_bounds(self.shape, i, j, k):
                type = 1  # Mark as source
                input_pressures = self.get_field(center=center, point=voxel, rotation_matrix=rotation_matrix, sound_speed=self.sound_speed, density=self.density)

                soxel = Soxel(
                    idx=self.source_config.idx,
                    type = type,
                    input_pressures = input_pressures,
                    acoustic_shader = self.source_config.acoustic_shader
                )
                soxels.append([i,j,k,soxel])
        return soxels

    def _euler_to_rotation_matrix(self, euler_angles: Tuple[float, float, float]) -> np.ndarray:
        """
        Convert Euler angles (x, y, z) to rotation matrix.
        Uses ZYX convention (yaw, pitch, roll).
        """
        rx, ry, rz = np.radians(euler_angles)
        
        # Rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix (ZYX convention)
        rotation_matrix = Rz @ Ry @ Rx
        return rotation_matrix

    def _rotate_vertices(self, vertices: List[Tuple[float, float, float]], rotation_matrix: np.ndarray, center: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Rotate vertices around center using rotation matrix"""
        center_array = np.array(center)
        rotated_vertices = []
        
        for vertex in vertices:
            vertex_array = np.array(vertex)
            # Translate to origin, rotate, then translate back
            translated_vertex = vertex_array - center_array
            rotated_vertex = rotation_matrix @ translated_vertex
            final_vertex = rotated_vertex + center_array
            rotated_vertices.append(tuple(final_vertex))
            
        return rotated_vertices

    def _rotate_vector(self, vector: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        """Apply rotation matrix to a vector"""
        return rotation_matrix @ vector

    def _get_planar_voxels(self, vertices: List[Tuple[float, float, float]]):
        """
        Get vortexs for verteces
        """
        voxelized_vertex = []
        for vertex in vertices:
            i,j,k = _world_to_grid(self.voxel_size, self.grid_geometry, vertex)
            if _is_in_bounds(self.shape, i,j,k):
                voxelized_vertex.append([i,j,k])

        # Calculate bounding box
        min_x, min_y, min_z = np.min(voxelized_vertex, axis=0)
        max_x, max_y, max_z = np.max(voxelized_vertex, axis=0)

        # voxelize bounding box vertex

        voxels = []
        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                for z in range(min_z, max_z+1):
                    if self._is_point_on_polygon([x,y,z], voxelized_vertex):
                        voxels.append([x,y,z])
        return voxels

    def _is_point_on_polygon(self, point, vertices, tolerance=0.5):
        """
        Check if a point lies on the plane of a polygon and within its boundaries.

        Parameters:
        point (tuple/list): (x, y, z) coordinates of the point to check
        vertices (list): List of vertices [(x1,y1,z1), (x2,y2,z2), ...] defining the polygon
        tolerance (float): Numerical tolerance for floating point comparisons

        Returns:
        bool: True if point is on the polygon, False otherwise
        """

        # Convert to numpy arrays for easier calculations
        point = np.array(point)
        vertices = np.array(vertices)

        # Step 1: Check if point lies on the polygon's plane
        if len(vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")

        # Calculate plane normal using first 3 vertices
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        normal = np.cross(v1, v2)

        # Check if point is on the plane
        vector_to_point = point - vertices[0]
        distance_to_plane = np.abs(np.dot(vector_to_point, normal))

        if distance_to_plane > tolerance:
            return False  # Point not on the polygon's plane

        # Step 2: Project point and vertices to 2D (using the polygon's plane)
        # Find the dominant axis to project onto
        abs_normal = np.abs(normal)
        if abs_normal[0] >= abs_normal[1] and abs_normal[0] >= abs_normal[2]:
            # Project onto YZ plane
            proj_point = point[1:3]
            proj_vertices = vertices[:, 1:3]
        elif abs_normal[1] >= abs_normal[0] and abs_normal[1] >= abs_normal[2]:
            # Project onto XZ plane
            proj_point = np.array([point[0], point[2]])
            proj_vertices = vertices[:, [0, 2]]
        else:
            # Project onto XY plane
            proj_point = point[0:2]
            proj_vertices = vertices[:, 0:2]

        # Step 3: Use ray casting algorithm to check if point is inside polygon
        return self._is_point_in_polygon_2d(proj_point, proj_vertices, tolerance)

    def _is_point_in_polygon_2d(self, point, vertices, tolerance=1e-6):
        """
        Check if a 2D point is inside a polygon using ray casting algorithm.
        """
        x, y = point
        n = len(vertices)
        inside = False

        p1x, p1y = vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n]

            # Check if point is on an edge
            if self._is_point_on_line_segment((x, y), (p1x, p1y), (p2x, p2y), tolerance):
                return True

            # Ray casting algorithm
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _is_point_on_line_segment(self, point, line_start, line_end, tolerance=1e-6):
        """
        Check if a point lies on a line segment.
        """
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Check if point is collinear with line segment
        cross_product = abs((px - x1) * (y2 - y1) - (py - y1) * (x2 - x1))
        if cross_product > tolerance:
            return False

        # Check if point is between the endpoints
        dot_product = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)
        if dot_product < 0:
            return False

        squared_length = (x2 - x1)**2 + (y2 - y1)**2
        if dot_product > squared_length:
            return False

        return True

    def get_field(self, center: Tuple[int, int, int], point: Tuple[int, int, int], 
                 rotation_matrix: np.ndarray, sound_speed: float = None, density: float = None) -> AcousticField:
        """
        Compute acoustic pressure and velocity vectors at the boundary of a planar sound source with directional orientation.

        Parameters:
        -----------
        center: Tuple[int, int, int]
            Center of plane
        point: Tuple[int, int, int]
            Coordinate of point on plane surface
        rotation_matrix: np.ndarray
            3x3 rotation matrix for directional orientation
        sound_speed : float
            Sound speed of medium on boundary
        density : float
            Density of medium on boundary

        Returns:
        --------
        AcousticField: AcousticField
        """
        current_frame = self.frames.get()

        if sound_speed == None:
           sound_speed = self.sound_speed
        if density == None:
           density = self.density

        # carthesian to azimuth and elevation in local coordinates
        local_vector = np.array(point) - np.array(center)
        
        # Apply rotation to get global coordinates
        global_vector = self._rotate_vector(local_vector, rotation_matrix)
        
        x, y, z = global_vector
        azimuth, elevation, r = _cartesian_to_spherical(x, y, z)

        fd_field = AcousticField([])
        for index in range(len(self.fd_samples)):
            low_freq = self.fd_samples[index][0]
            high_freq = self.fd_samples[index][1]
            sample = self.fd_samples[index][2][current_frame]

            magnitude_coeff = self.source_config.spatial_freq_response.get_avg_magnitude(azimuth, elevation, low_freq, high_freq)
            phase_coeff = self.source_config.spatial_freq_response.get_avg_phase(azimuth, elevation, low_freq, high_freq)

            # Apply both magnitude and phase coefficients
            # Treat the sample as a complex phasor for proper phase manipulation
            complex_sample = sample * magnitude_coeff * np.exp(1j * phase_coeff)
   
            # Use real part for acoustic pressure calculation
            sample = np.real(complex_sample)

            pressure, velocity = self._compute_acoustic_fields(low_freq, high_freq, sample, self.sound_speed, self.density)
            
            # Rotate velocity vectors to match source orientation
            velocity_rotated = self._rotate_vector(velocity, rotation_matrix)
            velocity_vectors = VelocityVectors(velocity_rotated[0], velocity_rotated[1], velocity_rotated[2])
            
            fd_field.add_field(low_freq=low_freq, high_freq=high_freq, pressure=pressure, velocity=velocity_vectors)
        return fd_field

    def _compute_acoustic_fields(self, low_freq: float, high_freq: float, sample: float, sound_speed: float, density: float) -> Tuple[float, np.ndarray]:
        """
        Compute acoustic pressure and velocity vectors at the boundary of a planar sound source.
    
        Parameters:
        -----------
        low_freq : float
            Lower frequency bound for filtering (Hz)
        high_freq : float
            Upper frequency bound for filtering (Hz)
        sample : float
            Spatialtial Frequency Responce aware sound sample
        sound_speed : float
            Speed of sound (m/s), default 343.0
        density : float
            Density (kg/m³), default 1.225
    
        Returns:
        --------
        pressure : np.ndarray
            Acoustic pressure boundary (Pa)
        velocity_vectors : np.ndarray
            Velocity vectors [(vx, vy, vz)] (m/s)
        """
        plane_vertices = self.source_config.geometry

        # Compute acoustic pressure for planar source
        if plane_vertices is None:
            # Infinite plane approximation
            # For an infinite plane, pressure = ρc * v (plane wave relationship)
            surface_velocity = sample
            pressure = density * sound_speed * surface_velocity
        else:
            # Finite polygonal plane
            plane_vertices = np.array(plane_vertices)
        
            # Calculate plane properties
            if len(plane_vertices) < 3:
                raise ValueError("Plane vertices must contain at least 3 points")
        
            # Calculate plane normal vector
            v1 = plane_vertices[1] - plane_vertices[0]
            v2 = plane_vertices[2] - plane_vertices[0]
            normal_vector = np.cross(v1, v2)
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
        
            # Calculate plane area (for finite size effects)
            area = self._calculate_polygon_area(plane_vertices)
        
            # For a finite planar source, we use a simplified model
            # Pressure depends on distance from center and frequency
            center_point = np.mean(plane_vertices, axis=0)
        
            # Use the filtered audio as surface velocity
            surface_velocity = sample
        
            # Calculate wave number for center frequency
            center_freq = np.sqrt(low_freq * high_freq)
            k = 2 * np.pi * center_freq / sound_speed
        
            # Simplified finite planar source model
            # This accounts for diffraction effects at edges
            characteristic_size = np.sqrt(area)
            ka = k * characteristic_size
        
            if ka < 1:
                # Small planar source - approaches point source behavior
                pressure = density * sound_speed * surface_velocity
            else:
                # Larger planar source - modified impedance
                # This is a simplified model - more sophisticated approaches would use
                # boundary element methods or Rayleigh integral
                Z = density * sound_speed * (1 + 0.5j / (ka))  # Approximate impedance
                pressure = np.real(Z * surface_velocity)
    
        # Compute velocity vectors
        # For a planar source, velocity is normal to the surface
        velocity_vectors = [0,0,0]
    
        if plane_vertices is None:
            # Infinite plane - velocity purely in z-direction (assuming plane at z=0)
            velocity_magnitude = np.abs(pressure) / (density * sound_speed)
            velocity_vectors[2] = velocity_magnitude * np.sign(pressure)
        else:
            # Finite polygonal plane - velocity normal to the plane
            normal_vector = self._calculate_plane_normal(plane_vertices)
            velocity_magnitude = np.abs(pressure) / (density * sound_speed)
        
            # Velocity direction follows the normal vector, magnitude varies with pressure
            velocity_vectors = normal_vector * velocity_magnitude * np.sign(pressure)
    
        return pressure, velocity_vectors

    def _calculate_plane_normal(self, vertices: np.ndarray) -> np.ndarray:
        """Calculate normal vector for a planar polygon."""
        if len(vertices) < 3:
            raise ValueError("Need at least 3 vertices to define a plane")
    
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        normal = np.cross(v1, v2)
        return normal / np.linalg.norm(normal)

    def _calculate_polygon_area(self, vertices: np.ndarray) -> float:
        """Calculate area of a planar polygon using shoelace formula."""
        if len(vertices) < 3:
            return 0.0
    
        # Project vertices to 2D by finding the best-fit plane
        centroid = np.mean(vertices, axis=0)
        centered_vertices = vertices - centroid
    
        # Use PCA to find the principal plane
        _, _, vh = np.linalg.svd(centered_vertices)
        normal = vh[2]  # Third component is the normal
    
        # Project vertices onto the plane
        if abs(normal[2]) > 1e-10:
            # Simple projection if not already aligned with XY plane
            basis1 = vh[0]
            basis2 = vh[1]
            proj_vertices = np.column_stack([
                np.dot(centered_vertices, basis1),
                np.dot(centered_vertices, basis2)
            ])
        else:
            proj_vertices = vertices[:, :2]
    
        # Shoelace formula
        x = proj_vertices[:, 0]
        y = proj_vertices[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
        return area
