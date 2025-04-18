import math
import mesh2sdf
import trimesh
from trimesh import Trimesh
import numpy as np
from skimage.measure import marching_cubes
import os


# File control

def make_dir_if_not_exist(path):
    if(not os.path.exists(path)):
        print("Folder does not exist, creating the folder: " + path)
        os.mkdir(path)


# Trimesh

def mesh_to_sdf_tensor(mesh: Trimesh, resolution:int = 64, recenter: bool = True, scaledownFactor = 1):
    """Creates a normalized signed distance function from a provided mesh, using a voxel grid

    Args:
        mesh (Trimesh): The mesh to convert, can be (non) watertight
        resolution (int, optional): the voxel resolution. Defaults to 64.

    Returns:
        sdf, mesh: the (res, res, res) np.array sdf and the fixed mesh
    """

    # normalize mesh
    vertices = mesh.vertices
    if(recenter):
        center = mesh.centroid
    else : center = 0
    scale = 2 /  np.max(mesh.extents) * scaledownFactor
    vertices = (vertices - center) * scale

    # fix mesh
    sdf, sdf_mesh = mesh2sdf.compute(
        vertices, mesh.faces, resolution, fix=(not mesh.is_watertight), level=2 / resolution, return_mesh=True)
    
    sdf_mesh.vertices = mesh.vertices / scale + center
    mesh.vertices = vertices /2
    return sdf, mesh

def get_point_colors_trimesh(mesh, points, filter = None):
    if(filter is not None):
        filteredPoints = points[filter,:]
    else: filteredPoints = points
    # get the indexes [n] and coordinate [n,3] of the closest triangle and point for each sample point
    closestPoints,_,triangleIds =  trimesh.proximity.closest_point(mesh, filteredPoints)
    # get the 3 vertex indices of each triangle [n,3]
    faces = mesh.faces[triangleIds]
    # get the uv coordinate of each vertex [n,3,2]
    uvCoordinates = mesh.visual.uv[faces]
    # get the barycentric coordinate of the closest point
    bary_coords = trimesh.triangles.points_to_barycentric(triangles=mesh.vertices[faces], points=closestPoints)
    # Interpolate UV coordinates using barycentric weights
    uv_points = np.einsum("ij,ijk->ik", bary_coords, uvCoordinates)
    # get uv color of each uv center [n,4]
    pointColors = mesh.visual.material.to_color(uv_points)
    if(filter is not None):
        allPointcolors = np.repeat([[0,0,0,0]], points.shape[0], 0)
        allPointcolors[filter] = pointColors
        return allPointcolors
    return pointColors


def sdf_to_mesh(sdf, spacing, center = True):
    vertices, faces, normals, _ = marching_cubes(sdf, level=0.0, spacing=(spacing,spacing, spacing))
    # Create a Trimesh object
    if(center):
        vertices = vertices - np.array([0.5,0.5,0.5])
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return new_mesh


def create_voxel_grid(size, center = True):
    # Set the starting point of the grid
    if(center):
        start = -0.5
    else:
        start = 0

    voxel_coordinates = np.linspace(start + 1/size/2, start + 1 - (1/size/2), size)  # n evenly spaced points in [0, 1]
    x, y, z = np.meshgrid(voxel_coordinates, voxel_coordinates, voxel_coordinates, indexing='ij')
    # Stack the coordinates into an (n x n x n x 3) array
    voxel_grid = np.stack((x, y, z), axis=-1)
    return voxel_grid

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh

def create_tiled_texture(sdf_distances: np.ndarray, sdf_colors: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
    """
    Creates a 2D texture by arranging slices of a 3D Signed Distance Field (SDF) along the Z-axis in a grid.
    
    Parameters:
        sdf_distances (np.ndarray): A 3D array of shape (H, W, D) representing the Signed Distance Field values,
                                     where H is height, W is width, and D is the depth (number of Z-axis slices).
        sdf_colors (np.ndarray): A 3D array of shape (H, W, D, 4) representing the RGBA colors for each slice.
        grid_size (tuple[int, int]): A tuple (grid_rows, grid_cols) representing the number of rows and columns 
                                      to arrange the slices into a grid.
    
    Returns:
        np.ndarray: A 2D array of shape (grid_rows * H, grid_cols * W, 4) representing the arranged texture,
                    where each slice is correctly placed in the grid and rotated as needed.
    """
    H, W, D = sdf_distances.shape  # H: Height, W: Width, D: Depth (Z-axis slices)
    
    # Create a copy of sdf_distances to avoid modifying the original
    dist = sdf_distances.copy()

    # Apply thresholding for visualization: 
    # - Set positive distances to 0 (transparent)
    # - Set non-positive distances to 255 (opaque)
    dist[sdf_distances <= 0] = 255
    dist[sdf_distances > 0] = 0
    
    # Ensure sdf_colors are in the range [0, 255] and take only the RGB channels (ignoring alpha for now)
    sdf_colors = (sdf_colors).astype(np.uint8)[:, :, :, :3]
    
    # Create RGBA frames by combining colors and distances
    rgba_frames = np.concatenate([sdf_colors, dist[..., None]], axis=-1)
    
    # Reverse the slices along the Z-axis (depth dimension)
    rgba_frames = rgba_frames[:, :, ::-1, :]  # Reverse the slices along the 3rd axis (Z-axis)
    
    # Unpack grid dimensions
    grid_rows, grid_cols = grid_size
    
    # Ensure that the grid is large enough to fit all slices
    assert grid_rows * grid_cols >= D, "Grid size must be large enough to fit all slices"
    
    # Calculate the final texture size based on grid dimensions
    flipbook_height = grid_rows * H
    flipbook_width = grid_cols * W
    
    # Create an empty texture to store the final grid of slices
    flipbook_texture = np.zeros((flipbook_height, flipbook_width, 4), dtype=np.uint8)
    
    # Place each slice into the grid, rotating and flipping as needed
    for idx in range(D):
        row = idx // grid_cols
        col = idx % grid_cols
        y, x = row * H, col * W
        # Place each slice in the correct position in the texture grid
        flipbook_texture[y:y+H, x:x+W, :] = np.transpose(rgba_frames[:, :, idx], (1, 0, 2))
        flipbook_texture[y:y+H, x:x+W, :] = np.flipud(flipbook_texture[y:y+H, x:x+W, :])  # Flip vertically
    
    return flipbook_texture



def get_closest_factors(input_value: int):
    if input_value <= 0:
        raise ValueError("Input must be greater than 0")
    
    test_num = int(math.sqrt(input_value))
    while input_value % test_num != 0:
        test_num -= 1
    
    return (test_num, input_value // test_num)