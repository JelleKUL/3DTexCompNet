import os
import numpy as np
import trimesh
from tqdm import tqdm
import utils as ut  # your own tools
from pathlib import Path
import matplotlib.pyplot as plt


def process_shapenet_model(mesh_path, save_dir, voxel_resolution=64):
    mesh = trimesh.load(mesh_path, force='mesh')
    mesh_name = Path(mesh_path).parent.parent.stem
    save_path = os.path.join(save_dir, mesh_name + "_full.npz")

    try:
        # Compute SDF and normalize
        sdf, normalized_mesh = ut.mesh_to_sdf_tensor(
            ut.as_mesh(mesh),
            voxel_resolution,
            recenter=True,
            scaledownFactor=0.85
        )

        # Get voxel color grid (RGBA)
        voxel_grid = ut.create_voxel_grid(voxel_resolution, True)
        mask = (np.array(sdf) <= 0.1).reshape(-1)
        colors = ut.get_point_colors_trimesh(
            normalized_mesh,
            voxel_grid.reshape(-1, 3),
            filter=mask
        ).reshape(voxel_resolution, voxel_resolution, voxel_resolution, 4)

        # Transpose to channels-first (4, D, H, W)
        voxel_tensor = np.transpose(colors, (3, 0, 1, 2)).astype(np.float32)
        voxel_tensor[3] = sdf  # Replace alpha channel with true SDF

        # Save
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(save_path, data=voxel_tensor)
        return True

    except Exception as e:
        print(f"Failed to process {mesh_path}: {e}")
        return False

def get_files_with_extension(folderPath,extension = ".obj", max_models = -1):
    input_list = []
    for subdir, dirs, files in os.walk(folderPath):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(extension):
                #print (filepath)
                input_list.append(filepath)
                if(max_models > 0):
                    if(len(input_list) >= max_models):
                        return input_list
    return input_list

def batch_process_shapenet(root_dir, save_dir, category_id='03001627', max_models=None):
    """
    Processes all .obj models in ShapeNet category and saves them as voxel tensors.
    """
    category_path = os.path.join(root_dir, category_id)
    obj_files = get_files_with_extension(category_path, max_models=max_models)
    print(f"Found {len(obj_files)} .obj files under category {category_id}")

    if max_models:
        obj_files = obj_files[:max_models]

    print(f"Processing {len(obj_files)} .obj files")

    for path in obj_files:
        process_shapenet_model(str(path), save_dir)
    #for path in tqdm(obj_files):
    #    process_shapenet_model(str(path), save_dir)


def visualize_voxel_tensor(npz_path, sdf_threshold=0.0):
    """
    Visualizes a 3D texture (RGB + SDF) stored in a .npz file.

    Args:
        npz_path (str): Path to the .npz file.
        sdf_threshold (float): Iso-surface value for rendering surface voxels.
    """
    data = np.load(npz_path)['data']  # shape: (4, D, H, W)
    rgb = data[:3]  # (3, D, H, W)
    sdf = data[3]   # (D, H, W)

    # Create occupancy mask (surface voxels only)
    mask = sdf <= sdf_threshold

    if not np.any(mask):
        print("No surface voxels found to visualize.")
        return

    # Get voxel indices
    x, y, z = np.where(mask)

    # Normalize RGB to [0, 1]
    rgb = np.clip(rgb, 0, 1)

    # Gather colors at voxel positions
    colors = np.stack([rgb[0][mask], rgb[1][mask], rgb[2][mask]], axis=1)

    # Plot using matplotlib 3D voxels
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=colors, marker='o', s=2)
    ax.set_title(f"Voxel Visualization: {Path(npz_path).stem}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()
