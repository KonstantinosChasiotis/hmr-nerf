# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from typing import Callable
import torch
from torch.utils.data import Dataset
from .parse_data import load_nerf_standard_data


class MultiviewDataset(Dataset):
    """This is a static multiview image dataset class.

    This class should be used for training tasks where the task is to fit a static 3D volume from
    multiview images.
    """

    def __init__(self, 
        dataset_path             : str,
        mip                      : int      = None,
        bg_color                 : str      = None,
        sample_rays              : bool     = False,
        n_rays                   : int      = 1024,
        split                    : str      = 'train',
        **kwargs
    ):
        """Initializes the dataset class.

        Note that the `init` function to actually load images is separate right now, because we don't want 
        to load the images unless we have to. This might change later.

        Args: 
            dataset_path (str): Path to the dataset.
            multiview_dataset_format (str): The dataset format. Currently supports standard (the same format
                used for instant-ngp) and the RTMV dataset.
            mip (int): The factor at which the images will be downsampled by to save memory and such.
                       Will downscale by 2**mip.
            bg_color (str): The background color to use for images with 0 alpha.
            dataset_num_workers (int): The number of workers to use if the dataset format uses multiprocessing.
        """
        self.root = dataset_path
        self.mip = mip
        self.bg_color = bg_color
        self.sample_rays = sample_rays
        self.n_rays = n_rays
        self.split = split
        self.init()

    def init(self):
        """Initializes the dataset.
        """

        # Get image tensors 
        
        self.coords = None

        self.data = self.get_images(self.split, self.mip)

        self.img_shape = self.data["imgs"].shape[1:3]
        self.num_imgs = self.data["imgs"].shape[0]

        self.data["imgs"] = self.data["imgs"].reshape(self.num_imgs, -1, 3)
        self.data["rays"] = self.data["rays"].reshape(self.num_imgs, -1, 6)
        if "masks" in self.data:
            self.data["masks"] = self.data["masks"].reshape(self.num_imgs, -1, 1)

    def get_images(self, split='train', mip=None):
        """Will return the dictionary of image tensors.

        Args:
            split (str): The split to use from train, val, test
            mip (int): If specified, will rescale the image by 2**mip.

        Returns:
            (dict of torch.FloatTensor): Dictionary of tensors that come with the dataset.
        """
        
        data = load_nerf_standard_data(self.root, split,
                        bg_color=self.bg_color, num_workers=-1, mip=self.mip)

        return data

    def sample(self, inputs, num_samples):
        """ Samples a subset of rays from a single image.
            50% of the rays are sampled randomly from the image.
            50% of the rays are sampled randomly within the valid mask.
        """
        valid_idx = torch.nonzero(inputs['masks'].squeeze()).squeeze()

    
        ray_idx = torch.randperm(
            inputs['imgs'].shape[0],
            device=inputs['imgs'].device)[:num_samples]
        
        select_idx = torch.randperm( valid_idx.shape[0], device=inputs['imgs'].device) [:num_samples // 2] 

        ray_idx [:num_samples // 2] = valid_idx [select_idx]    
        out = {}
        out['rays'] = inputs['rays'][ray_idx].contiguous()
        out['imgs'] = inputs['imgs'][ray_idx].contiguous()
        out['masks'] = inputs['masks'][ray_idx].contiguous()
        return out


    def __len__(self):
        """Length of the dataset in number of rays.
        """
        return self.data["imgs"].shape[0]

    def __getitem__(self, idx : int):
        """Returns rays, gt ray colors, and binary masks. 
        """
        out = {}
        out['rays'] = self.data["rays"][idx].float()
        out['imgs'] = self.data["imgs"][idx].float()
        out['masks'] = self.data["masks"][idx].bool()

        if self.sample_rays and self.split == 'train':
            out = self.sample(out, self.n_rays)
        
        return out
    
    def get_img_samples(self, idx, batch_size):
        """Returns a batch of samples from an image, indexed by idx.
        """

        ray_idx = torch.randperm(self.data["imgs"].shape[1])[:batch_size]

        out = {}
        out['rays'] = self.data["rays"][idx, ray_idx]
        out['imgs'] = self.data["imgs"][idx, ray_idx]
        
        return out
