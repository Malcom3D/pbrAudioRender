import bpy
import bmesh
from pbrAudioCommon import np
from mathutils import Vector, Matrix

import sys, os
from pbrAudioCommon import np
sys.path.append(os.getcwd())
from typing import List, Dict, Any, Optional, Tuple, Union

from lib.frames import FrameCounter
from core.entity_manager import EntityManager
from core.soxel_grid import SoxelGrid
from utils.config import Config

config_file = 'config.json'
config = Config(config_file)

frames = FrameCounter() if config.system.frame_limit == None else FrameCounter(frame_limit=config.system.frame_limit)
entity_manager = EntityManager(config, frames)
soxel_grid = SoxelGrid(config, entity_manager, frames)

"""
Convert a 3D numpy array to a a voxel grid in Blender.

Parameters:
- voxel_array: 3D numpy array where non-zero values represent solid voxels
- voxel_size: Size of each voxel cube
- location: Location to place the voxel grid
- material: Optional material to apply to all voxels
"""
    
# Clear existing mesh objects (optional - remove if you want to keep existing objects)
# bpy.ops.object.select_all(action='SELECT')
# bpy.ops.object.delete()

voxel_size = config.acoustic_domain.voxel_size
location = config.acoustic_domain.geometry[0]    
# Get the dimensions of the voxel array
z_dim, y_dim, x_dim = config.acoustic_domain.shape
    
# Create a new mesh and object
mesh_name = "VoxelGrid"
obj_name = "VoxelGrid"

mesh = bpy.data.meshes.new(mesh_name)
obj = bpy.data.objects.new(obj_name, mesh)
    
# Link object to scene
bpy.context.collection.objects.link(obj)

# Create bmesh instance
bm = bmesh.new()

# Create cubes for each voxel
for z in range(z_dim):
    for y in range(y_dim):
        for x in range(x_dim):
            if soxel_grid.soxels[z, y, x].type != 0:  # Only create voxels for non-zero values
                # Calculate position
                pos_x = location[0] + x * voxel_size
                pos_y = location[1] + y * voxel_size
                pos_z = location[2] + z * voxel_size
                
                # Create a cube at this position
                bmesh.ops.create_cube(
                    bm,
                    size=voxel_size,
                    matrix=Matrix.Translation((pos_x, pos_y, pos_z))
                )
    
# Convert bmesh to mesh
bm.to_mesh(mesh)
bm.free()
        
# Select the object
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
