# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import glob
import cv2
import skimage
import imageio
import json
from tqdm import tqdm
import skimage.metrics
import logging as log
import numpy as np
import torch
from .camera.camera import Camera
from .camera.coordinates import blender_coords
from .camera.intrinsics import CameraFOV
""" A module for loading data files in the standard NeRF format, including extensions to the format
    supported by Instant Neural Graphics Primitives.
    See: https://github.com/NVlabs/instant-ngp
"""

################################ Load Image Function ########################################


def _resize_mip(img, mip, interpolation=cv2.INTER_LINEAR):
    """Resize image with cv2.

    Args:
        img (torch.FloatTensor): Image of shape [H, W, 3]
        mip (int): Rescaling factor. Will rescale by 2**mip.
        interpolation: Interpolation modes used by `cv2.resize`.

    Returns:
        (torch.FloatTensor): Rescaled image of shape [H/(2**mip), W/(2**mip), 3]
    """
    resize_factor = 2**mip
    # WARNING: cv2 expects (w,h) for the shape. God knows why :)
    shape = (int(img.shape[1] // resize_factor), int(img.shape[0] // resize_factor))
    img = cv2.resize(img, dsize=shape, interpolation=interpolation)
    return img


# Local function for multiprocess. Just takes a frame from the JSON to load images and poses.
def _load_standard_imgs(frame, root, mip=None):
    """

    Args:
        root: The root of the dataset.
        frame: The frame object from the transform.json.
        mip: If set, rescales the image by 2**mip.

    Returns:
        (dict): Dictionary of the image and pose.
    """
    fpath = os.path.join(root, frame['file_path'].replace("\\", "/"))

    basename = os.path.basename(os.path.splitext(fpath)[0])
    if os.path.splitext(fpath)[1] == "":
        # Assume PNG file if no extension exists... the NeRF synthetic data follows this convention.
        fpath += '.png'

    # For some reason instant-ngp allows missing images that exist in the transform but not in the data.
    # Handle this... also handles the above case well too.
    if os.path.exists(fpath):
        img = imageio.imread(fpath)
        img = skimage.img_as_float32(img)
        if mip is not None:
            img = _resize_mip(img, mip, interpolation=cv2.INTER_AREA)
        return dict(basename=basename,
                    img=torch.FloatTensor(img), pose=torch.FloatTensor(np.array(frame['transform_matrix'])))
    else:
        # log.info(f"File name {fpath} doesn't exist. Ignoring.")
        return None

################################ Ray Sampling Function ########################################


def _generate_default_grid(width, height, device=None):
    h_coords = torch.arange(height, device=device)
    w_coords = torch.arange(width, device=device)
    return torch.meshgrid(h_coords, w_coords, indexing='ij')  # return pixel_y, pixel_x


def _generate_centered_pixel_coords(img_width, img_height, res_x=None, res_y=None, device=None):
    pixel_y, pixel_x = _generate_default_grid(res_x, res_y, device)
    scale_x = 1.0 if res_x is None else float(img_width) / res_x
    scale_y = 1.0 if res_y is None else float(img_height) / res_y
    pixel_x = pixel_x * scale_x + 0.5   # scale and add bias to pixel center
    pixel_y = pixel_y * scale_y + 0.5   # scale and add bias to pixel center
    return pixel_y, pixel_x


# -- Ray gen --

def _to_ndc_coords(pixel_x, pixel_y, camera):
    pixel_x = 2 * (pixel_x / camera.width) - 1.0
    pixel_y = 2 * (pixel_y / camera.height) - 1.0
    return pixel_x, pixel_y


def _generate_pinhole_rays(camera: Camera, coords_grid: torch.Tensor):
    """Default ray generation function for pinhole cameras.

    This function assumes that the principal point (the pinhole location) is specified by a 
    displacement (camera.x0, camera.y0) in pixel coordinates from the center of the image. 

    The Kaolin camera class does not enforce a coordinate space for how the principal point is specified,
    so users will need to make sure that the correct principal point conventions are followed for 
    the cameras passed into this function.

    Args:
        camera (kaolin.render.camera): The camera class. 
        coords_grid (torch.FloatTensor): Grid of coordinates of shape [H, W, 2].

    Returns:
        (wisp.core.Rays): The generated pinhole rays for the camera.
    """
    if camera.device != coords_grid[0].device:
        raise Exception(f"Expected camera and coords_grid[0] to be on the same device, but found {camera.device} and {coords_grid[0].device}.")
    if camera.device != coords_grid[1].device:
        raise Exception(f"Expected camera and coords_grid[1] to be on the same device, but found {camera.device} and {coords_grid[1].device}.")
    # coords_grid should remain immutable (a new tensor is implicitly created here)
    pixel_y, pixel_x = coords_grid
    pixel_x = pixel_x.to(camera.device, camera.dtype)
    pixel_y = pixel_y.to(camera.device, camera.dtype)

    # Account for principal point (offsets from the center)
    pixel_x = pixel_x - camera.x0
    pixel_y = pixel_y + camera.y0

    # pixel values are now in range [-1, 1], both tensors are of shape res_y x res_x
    pixel_x, pixel_y = _to_ndc_coords(pixel_x, pixel_y, camera)

    ray_dir = torch.stack((pixel_x * camera.tan_half_fov(CameraFOV.HORIZONTAL),
                           -pixel_y * camera.tan_half_fov(CameraFOV.VERTICAL),
                           -torch.ones_like(pixel_x)), dim=-1)

    ray_dir = ray_dir.reshape(-1, 3)    # Flatten grid rays to 1D array
    ray_orig = torch.zeros_like(ray_dir)

    # Transform from camera to world coordinates
    ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)
    ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)
    ray_orig, ray_dir = ray_orig[0], ray_dir[0]  # Assume a single camera

    return torch.cat([ray_orig, ray_dir], dim=-1)


################################ Load NeRF Data Function ########################################


def load_nerf_standard_data(root, split='train', bg_color='white', num_workers=-1, mip=None):
    """Standard loading function.

    This follows the conventions defined in https://github.com/NVlabs/instant-ngp.

    There are two pairs of standard file structures this follows:

    ```
    /path/to/dataset/transform.json
    /path/to/dataset/images/____.png
    ```

    or

    ```
    /path/to/dataset/transform_{split}.json
    /path/to/dataset/{split}/_____.png
    ```

    Args:
        root (str): The root directory of the dataset.
        split (str): The dataset split to use from 'train', 'val', 'test'.
        bg_color (str): The background color to use for when alpha=0.
        num_workers (int): The number of workers to use for multithreaded loading. If -1, will not multithread.
        mip: If set, rescales the image by 2**mip.

    Returns:
        (dict of torch.FloatTensors): Different channels of information from NeRF.
    """

    transforms = sorted(glob.glob(os.path.join(root, "*.json")))

    transform_dict = {}

    train_only = False

    if mip is None:
        mip = 0

    if len(transforms) == 1:
        transform_dict['train'] = transforms[0]
        train_only = True
    elif len(transforms) == 3:
        fnames = [os.path.basename(transform) for transform in transforms]

        # Create dictionary of split to file path, probably there is simpler way of doing this
        for _split in ['test', 'train', 'val']:
            for i, fname in enumerate(fnames):
                if _split in fname:
                    transform_dict[_split] = transforms[i]
    else:
        assert False and "Unsupported number of splits, there should be ['test', 'train', 'val']"

    if split not in transform_dict:
        assert False and f"Split type ['{split}'] unsupported in the dataset provided"

    for key in transform_dict:
        with open(transform_dict[key], 'r') as f:
            transform_dict[key] = json.load(f)

    imgs = []
    poses = []
    basenames = []

    for frame in tqdm(transform_dict[split]['frames'], desc='loading data'):
        _data = _load_standard_imgs(frame, root, mip=mip)
        if _data is not None:
            basenames.append(_data["basename"])
            imgs.append(_data["img"])
            poses.append(_data["pose"])

    imgs = torch.stack(imgs)
    poses = torch.stack(poses)

    h, w = imgs[0].shape[:2]

    if 'x_fov' in transform_dict[split]:
        # Degrees
        x_fov = transform_dict[split]['x_fov']
        fx = (0.5 * w) / np.tan(0.5 * float(x_fov) * (np.pi / 180.0))
        if 'y_fov' in transform_dict[split]:
            y_fov = transform_dict[split]['y_fov']
            fy = (0.5 * h) / np.tan(0.5 * float(y_fov) * (np.pi / 180.0))
        else:
            fy = fx
    elif 'fl_x' in transform_dict[split]:
        fx = float(transform_dict[split]['fl_x']) / float(2**mip)
        if 'fl_y' in transform_dict[split]:
            fy = float(transform_dict[split]['fl_y']) / float(2**mip)
        else:
            fy = fx
    elif 'camera_angle_x' in transform_dict[split]:
        # Radians
        camera_angle_x = transform_dict[split]['camera_angle_x']
        fx = (0.5 * w) / np.tan(0.5 * float(camera_angle_x))

        if 'camera_angle_y' in transform_dict[split]:
            camera_angle_y = transform_dict[split]['camera_angle_y']
            fy = (0.5 * h) / np.tan(0.5 * float(camera_angle_y))
        else:
            fy = fx
    else:
        fx = 0.0
        fy = 0.0

    # The principal point in wisp are always a displacement in pixels from the center of the image.
    x0 = 0.0
    y0 = 0.0
    # The standard dataset generally stores the absolute location on the image to specify the principal point.
    # Thus, we need to scale and translate them such that they are offsets from the center.
    if 'cx' in transform_dict[split]:
        x0 = (float(transform_dict[split]['cx']) / (2**mip)) - (w//2)
    if 'cy' in transform_dict[split]:
        y0 = (float(transform_dict[split]['cy']) / (2**mip)) - (h//2)

    offset = transform_dict[split]['offset'] if 'offset' in transform_dict[split] else [0 ,0 ,0]
    scale = transform_dict[split]['scale'] if 'scale' in transform_dict[split] else 1.0
    aabb_scale = transform_dict[split]['aabb_scale'] if 'aabb_scale' in transform_dict[split] else 1.0

    poses[..., :3, 3] /= aabb_scale
    poses[..., :3, 3] *= scale
    poses[..., :3, 3] += torch.FloatTensor(offset)

    # nerf-synthetic uses a default far value of 6.0
    default_far = 6.0

    rays = []

    cameras = dict()
    for i in tqdm(range(imgs.shape[0]), desc='generating rays'):
        view_matrix = torch.zeros_like(poses[i])
        view_matrix[:3, :3] = poses[i][:3, :3].T
        view_matrix[:3, -1] = torch.matmul(-view_matrix[:3, :3], poses[i][:3, -1])
        view_matrix[3, 3] = 1.0
        camera = Camera.from_args(view_matrix=view_matrix,
                                  focal_x=fx,
                                  focal_y=fy,
                                  width=w,
                                  height=h,
                                  far=default_far,
                                  near=0.0,
                                  x0=x0,
                                  y0=y0,
                                  dtype=torch.float64)
        camera.change_coordinate_system(blender_coords())
        cameras[basenames[i]] = camera
        ray_grid = _generate_centered_pixel_coords(camera.width, camera.height,
                                                  camera.width, camera.height, device='cpu')

        rays.append(
             _generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid).reshape(camera.height, camera.width, -1).to('cpu'))
   
    rays = torch.stack(rays)
    rgbs = imgs[... ,:3]
    alpha = imgs[... ,3:4]
    
    if alpha.numel() == 0:
        masks = torch.ones_like(rgbs[... ,0:1]).bool()
    else:
        masks = (alpha > 0.5).bool()

        if bg_color == 'black':
            rgbs[... ,:3] -= ( 1 -alpha)
            rgbs = np.clip(rgbs, 0.0, 1.0)
        else:
            rgbs[... ,:3] *= alpha
            rgbs[... ,:3] += ( 1 -alpha)
            rgbs = np.clip(rgbs, 0.0, 1.0)
    
    return {"imgs": rgbs, "masks": masks, "rays": rays, "cameras": cameras}
