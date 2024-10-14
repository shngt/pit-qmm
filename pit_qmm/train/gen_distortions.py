import open3d as o3d
import numpy as np
import random
import copy
import os
import torch
from pytorch3d.transforms import Rotate, Translate
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor
)
from pytorch3d.structures import Pointclouds
from PIL import Image
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Basic distortion functions
def add_gaussian_noise(pcd, mean=0.0, std=0.01):
    points = np.asarray(pcd.points)
    noise = np.random.normal(mean, std, points.shape)
    points += noise
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

# def add_poisson_noise(pcd, lam=1.0):
#     points = np.asarray(pcd.points)
#     noise = np.random.poisson(lam, points.shape)
#     points += noise
#     pcd.points = o3d.utility.Vector3dVector(points)
#     return pcd

def add_color_noise(pcd, mean=0.0, std=0.01):
    colors = np.asarray(pcd.colors)
    noise = np.random.normal(mean, std, colors.shape)
    colors = np.clip(colors + noise, 0, 1)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def gaussian_geometry_shifting(pcd, mean=0.0, std=0.01):
    return add_gaussian_noise(pcd, mean, std)

def high_frequency_noise(pcd, mean=0.0, std=0.01):
    return add_gaussian_noise(pcd, mean, std)

def uniform_geometry_shifting(pcd, low=-0.01, high=0.01):
    points = np.asarray(pcd.points)
    noise = np.random.uniform(low, high, points.shape)
    points += noise
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def quantization_noise(pcd, levels=256):
    points = np.asarray(pcd.points)
    points = np.round(points * levels) / levels
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def local_missing(pcd, percentage=0.1):
    points = np.asarray(pcd.points)
    num_points = points.shape[0]
    num_remove = int(num_points * percentage)
    indices = np.random.choice(num_points, num_remove, replace=False)
    points = np.delete(points, indices, axis=0)
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def mean_shift(pcd, shift=0.1):
    points = np.asarray(pcd.points)
    points += shift
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def local_offset(pcd, offset=0.1):
    points = np.asarray(pcd.points)
    indices = np.random.choice(points.shape[0], size=int(points.shape[0] * 0.1), replace=False)
    points[indices] += offset
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def change_contrast(pcd, factor=1.5):
    colors = np.asarray(pcd.colors)
    mean_color = np.mean(colors, axis=0)
    colors = mean_color + factor * (colors - mean_color)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def local_rotation(pcd, angle=np.pi/4):
    points = np.asarray(pcd.points)
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((angle, angle, angle))
    points = points @ rotation_matrix.T
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def change_color_saturation(pcd, factor=1.5):
    colors = np.asarray(pcd.colors)
    mean_color = np.mean(colors, axis=1, keepdims=True)
    colors = mean_color + factor * (colors - mean_color)
    colors = np.clip(colors, 0, 1)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def luminance_noise(pcd, mean=0.0, std=0.01):
    return add_color_noise(pcd, mean, std)

def spatially_correlated_noise(pcd, correlation=0.1):
    points = np.asarray(pcd.points)
    noise = np.random.normal(0, 1, points.shape)
    noise = correlation * np.roll(noise, 1, axis=0) + (1 - correlation) * noise
    points += noise
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def poisson_reconstruction(pcd):
    # Placeholder: Requires specific algorithms or external libraries
    print("Applying Poisson Reconstruction - Placeholder")
    return pcd

def multiplicative_gaussian_noise(pcd, mean=1.0, std=0.01):
    points = np.asarray(pcd.points)
    noise = np.random.normal(mean, std, points.shape)
    points *= noise
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

# Placeholder distortion functions for codec-based and complex distortions
def placeholder_distortion(pcd, distortion_name):
    print(f"Applying placeholder distortion: {distortion_name}")
    return pcd

# Dictionary of distortion functions
distortions = {
    "additive gaussian noise": add_gaussian_noise,
    # "Poisson noise": add_poisson_noise,
    "color noise": add_color_noise,
    # "Gaussian geometry shifting": gaussian_geometry_shifting,
    # "High frequency noise": high_frequency_noise,
    # "Uniform geometry shifting": uniform_geometry_shifting,
    "quantization noise": quantization_noise,
    # "Local missing": local_missing,
    # "Mean shift (intensity shift)": mean_shift,
    # "Local offset": local_offset,
    "contrast change": change_contrast,
    # "Local rotation": local_rotation,
    # "Change of color saturation": change_color_saturation,
    # "Luminance noise": luminance_noise,
    # "Spatially correlated noise": spatially_correlated_noise,
    # "Poisson Reconstruction": poisson_reconstruction,
    # "Multiplicative Gaussian noise": multiplicative_gaussian_noise,
    # "GPCC-lossless G and lossy A": placeholder_distortion,
    # "Color quantization with dither": placeholder_distortion,
    # "GPCC-lossless G and nearlossless A": placeholder_distortion,
    # "Octree Compression": placeholder_distortion,
    # "GPCC-lossy G and lossy A": placeholder_distortion,
    # "Down sample": downsample,
    # "VPCC-lossy G and lossy A": placeholder_distortion,
    # "Saltpepper noise": placeholder_distortion,
    # "AVS-limitlossy G and lossy A": placeholder_distortion,
    # "Rayleigh noise": placeholder_distortion,
    # "AVS-lossless G and limitlossy A": placeholder_distortion,
    # "Gamma noise": placeholder_distortion,
    # "AVS-lossless G and lossy A": placeholder_distortion,
    # "Uniform noise (white noise)": add_uniform_noise,
}

def apply_distortion(pcd, distortion_name):
    if distortion_name in distortions:
        return distortions[distortion_name](pcd)
    else:
        return placeholder_distortion(pcd, distortion_name)

def pc_norm(pc):
    """ pc: NxC, return NxC """
    xyz = np.asarray(pc.points)
    # other_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m if m != 0 else xyz

    pc.points = o3d.utility.Vector3dVector(xyz)
    return pc

def tensor_to_image(tensor: torch.Tensor, file_path: str):
    # Check if tensor is on the GPU and move it to CPU if necessary
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Check if the tensor has 3 dimensions (C, H, W)
    assert tensor.ndimension() == 3, "Tensor must be 3-dimensional (C, H, W)"
    # breakpoint()
    # Convert tensor to numpy array and scale it to [0, 255]
    array = tensor.numpy()
    array = (array * 255).astype(np.uint8)
    
    # Transpose the array to (H, W, C)
    array = array.transpose(1, 2, 0)
    
    # Convert the numpy array to a PIL image
    image = Image.fromarray(array)
    
    # Save the image
    image.save(file_path)
 
def multiview_projections(points):
    """
    Function to compute multiview projections of a 3D point cloud using PyTorch3D.
    Saves the projections as PNG files.
    Parameters:
    - points (torch tensor): Nx3 tensor of 3D point cloud coordinates.
    - output_folder (str): Folder path to save the PNG files.
    """
    # Initialize mesh with vertices and faces (none for point cloud)
    # verts = points.unsqueeze(0)  # add batch dimension
    # faces = None  # point cloud, no faces
 
    # Set up device (GPU if available, otherwise CPU)
 
    # # Set up lights
    # lights = PointLights(device=device, location=[[2.0, 2.0, -2.0]])
 
    images = []
    # Set up cameras with 6 views (cube faces)
    # Rs, Ts = look_at_view_transform(dist=2.5, elev=torch.tensor(), azim=torch.tensor(), device=device)
    for elev, azim in zip([0., 0., 0., 0., 90., -90.], [0., 90., 180., 270., 0., 0.]):
        R, T = look_at_view_transform(dist=2.5, elev=elev, azim=azim, device=device)
        # breakpoint()
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
        # Rasterization settings for images
        raster_settings = PointsRasterizationSettings(
            image_size=512, 
            radius = 0.003,
            points_per_pixel = 10,
            max_points_per_bin = 1000000
        )
    
        # Renderer for point cloud
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )

        images.append(renderer(points).squeeze(0).permute(2, 0, 1))

    return images

# Function to segment out an octant from the point cloud (with colors)
def segment_octant_with_colors(pcd, octant):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Determine the mask for the specified octant
    if octant == 1:
        mask = (points[:, 0] > 0) & (points[:, 1] > 0) & (points[:, 2] > 0)
    elif octant == 2:
        mask = (points[:, 0] < 0) & (points[:, 1] > 0) & (points[:, 2] > 0)
    elif octant == 3:
        mask = (points[:, 0] < 0) & (points[:, 1] < 0) & (points[:, 2] > 0)
    elif octant == 4:
        mask = (points[:, 0] > 0) & (points[:, 1] < 0) & (points[:, 2] > 0)
    elif octant == 5:
        mask = (points[:, 0] > 0) & (points[:, 1] > 0) & (points[:, 2] < 0)
    elif octant == 6:
        mask = (points[:, 0] < 0) & (points[:, 1] > 0) & (points[:, 2] < 0)
    elif octant == 7:
        mask = (points[:, 0] < 0) & (points[:, 1] < 0) & (points[:, 2] < 0)
    elif octant == 8:
        mask = (points[:, 0] > 0) & (points[:, 1] < 0) & (points[:, 2] < 0)
    else:
        raise ValueError("Octant must be between 1 and 8.")
    
    # Apply mask to extract points and colors in the selected octant
    octant_points = points[mask]
    octant_colors = colors[mask]
    
    # Create a new Open3D PointCloud object for the octant
    octant_pcd = o3d.geometry.PointCloud()
    octant_pcd.points = o3d.utility.Vector3dVector(octant_points)
    octant_pcd.colors = o3d.utility.Vector3dVector(octant_colors)
    
    return octant_pcd, mask

# Function to merge the modified octant back into the original point cloud
def merge_octant_back(pcd, modified_octant_pcd, mask):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    modified_points = np.asarray(modified_octant_pcd.points)
    modified_colors = np.asarray(modified_octant_pcd.colors)
    
    points[mask] = modified_points
    colors[mask] = modified_colors
    
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def get_conv(object_id, distortion, octant):
    conversation_template = [
        {
            "from": "human",
            "value": "<|image|><|image|><|image|><|image|><|image|><|image|>\n<point>\nCan you identify the nature and location of the distortion?"
        },
        {
            "from": "gpt",
            "value": f"The point cloud shows {distortion} in octant {octant}."
        }
    ]
    entry = {}
    entry['object_id'] = object_id
    entry['conversations'] = conversation_template
    return entry

# def main(input_file, output_dir):
#     pcd = o3d.io.read_point_cloud(input_file)

#     for distortion_name in distortions.keys():
#         pcd_copy = copy.deepcopy(pcd)
#         distorted_pcd = apply_distortion(pcd_copy, distortion_name)
#         output_file = f"{output_dir}/{distortion_name.replace(' ', '_')}.ply"
#         o3d.io.write_point_cloud(output_file, distorted_pcd)
#         print(f"Saved distorted point cloud: {output_file}")

# Function to convert Open3D point cloud to PyTorch3D Pointclouds object
def o3d_to_p3d(pcd):
    # Extract points as a numpy array
    points = np.asarray(pcd.points)
    
    # Convert points to a PyTorch tensor
    points_tensor = torch.tensor(points, dtype=torch.float32)
    
    # Check if the point cloud has colors
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        colors_tensor = torch.tensor(colors, dtype=torch.float32)
        point_cloud_p3d = Pointclouds(points=[points_tensor], features=[colors_tensor])
    else:
        point_cloud_p3d = Pointclouds(points=[points_tensor])
    
    point_cloud_p3d = point_cloud_p3d.to(device)

    return point_cloud_p3d

def distort_octant_with_type(pcd_copy, distortion_name, octant):
    pcd_copy = pc_norm(pcd_copy)
    octant_pcd, mask = segment_octant_with_colors(pcd_copy, octant=octant)

    # Apply a distortion
    distorted_octant_pcd = apply_distortion(octant_pcd, distortion_name)

    # Merge the modified octant back into the original point cloud
    merged_pcd = merge_octant_back(pcd_copy, distorted_octant_pcd, mask)

    return merged_pcd

if __name__ == "__main__":
    root_dir = '/proj/esv-summer-interns/home/eguhpas/samples_with_MOS/pristine'
    anno = []
    for f in os.listdir(root_dir):
        
        ply_file_path = os.path.join(root_dir, f)
        base_name = ply_file_path.split(os.sep)[-1].split('.')[0]
        if not f.endswith('ply'):
            continue

        pcd = o3d.io.read_point_cloud(ply_file_path)

        for distortion_name in distortions.keys():
            for octant in range(1, 9):
                distortion_filename = distortion_name.lower().replace(' ', '_')
                object_id = f'{base_name}_{distortion_filename}_{octant}'
                anno.append(get_conv(object_id, distortion_name.lower(), octant))
                if os.path.exists(f'lspcqa_syn_views/{object_id}'):
                    continue

                pcd_copy = copy.deepcopy(pcd)

                # Segment out the desired octant (e.g., first octant)
                merged_pcd = distort_octant_with_type(pcd_copy, distortion_name, octant)
                
                # output_file = f"{output_dir}/{distortion_name.replace(' ', '_')}.ply"
                # o3d.io.write_point_cloud(output_file, distorted_pcd)
                images = multiview_projections(o3d_to_p3d(merged_pcd))

                os.makedirs(f'lspcqa_syn_views/{object_id}', exist_ok=True)
                views = ['front', 'right', 'back', 'left', 'top', 'bottom']
                for view, image in zip(views, images):
                    tensor_to_image(image, f'lspcqa_syn_views/{object_id}/{object_id}_{view}.png')
                print(f'{object_id} processed!')
        
    with open('lspcqa_syn_oct_localization.json', 'w+') as f:
        json.dump(anno, f)