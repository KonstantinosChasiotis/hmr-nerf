import os
import logging as log
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import trimesh
import mcubes
import wandb

from .ray import exponential_integration
from ..utils.metrics import psnr

# Warning: you MUST NOT change the resolution of marching cube
RES = 256

class Trainer(nn.Module):

    def __init__(self, config, model, pe, log_dir):

        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"[INFO] Using device: {self.device}")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.cfg = config
        self.pos_enc = pe.to(self.device)
        self.mlp = model.to(self.device)
        self.log_dir = log_dir
        self.log_dict = {}

        self.init_optimizer()
        self.init_log_dict()

    def init_optimizer(self):
        
        trainable_parameters = list(self.mlp.parameters())
        self.optimizer = torch.optim.Adam(trainable_parameters, lr=self.cfg.lr, 
                                    betas=(self.cfg.beta1, self.cfg.beta2),
                                    weight_decay=self.cfg.weight_decay)

    def init_log_dict(self):
        """Custom log dict.
        """
        self.log_dict['total_loss'] = 0.0
        self.log_dict['rgb_loss'] = 0.0
        self.log_dict['total_iter_count'] = 0
        self.log_dict['image_count'] = 0


    def sample_points(self, ray_orig, ray_dir, near=1.0, far=3.0, num_points=64):
        """Sample points along rays. Retruns 3D coordinates of the points.
        TODO: One and extend this function to the hirachical sampling technique 
             used in NeRF or design a more efficient sampling technique for 
             better surface reconstruction.

        Args:
            ray_orig (torch.FloatTensor): Origin of the rays of shape [B, Nr, 3].
            ray_dir (torch.FloatTensor): Direction of the rays of shape [B, Nr, 3].
            near (float): Near plane of the camera.
            far (float): Far plane of the camera.
            num_points (int): Number of points (Np) to sample along the rays.

         Returns:
            points (torch.FloatTensor): 3D coordinates of the points of shape [B, Nr, Np, 3].
            z_vals (torch.FloatTensor): Depth values of the points of shape [B, Nr, Np, 1].
            deltas (torch.FloatTensor): Distance between the points of shape [B, Nr, Np, 1].

        """
        B, Nr = ray_orig.shape[:2]

        t = torch.linspace(0.0, 1.0, num_points, device=ray_orig.device).view(1, 1, -1) + \
            (torch.rand(B, Nr, num_points, device=ray_orig.device) / num_points)

        z_vals = near * (1.0 - t) + far * t
        points = ray_orig[:, :, None, :] + ray_dir[:, :, None, :] * z_vals[..., None]
        deltas = z_vals.diff(dim=-1, prepend=(torch.zeros(B, Nr, 1, device=z_vals.device)+ near))

        return points, z_vals[..., None], deltas[..., None]

    def predict_radience(self, coords):
        """Predict radiance at the given coordinates.
        TODO: You can adjust the network architecture according to your needs. You may also 
        try to use additional raydirections inputs to predict the radiance.

        Args:
            coords (torch.FloatTensor): 3D coordinates of the points of shape [..., 3].

        Returns:
            rgb (torch.FloatTensor): Radiance at the given coordinates of shape [..., 3].
            sigma (torch.FloatTensor): volume density at the given coordinates of shape [..., 1].

        """
        if len(coords.shape) == 2:
            coords = self.pos_enc(coords)
        else:
            input_shape = coords.shape
            coords = self.pos_enc(coords.view(-1, 3)).view(*input_shape[:-1], -1)

        pred = self.mlp(coords)
        rgb = torch.sigmoid(pred[..., :3])
        sigma = torch.relu(pred[..., 3:])

        return rgb, sigma

    def volume_render(self, rgb, sigma, depth, deltas):
        """Ray marching to compute the radiance at the given rays.
        TODO: You are free to try out different neural rendering methods.
        
        Args:
            rgb (torch.FloatTensor): Radiance at the sampled points of shape [B, Nr, Np, 3].
            sigma (torch.FloatTensor): Volume density at the sampled points of shape [B, Nr, Np, 1].
            deltas (torch.FloatTensor): Distance between the points of shape [B, Nr, Np, 1].
        
        Returns:
            ray_colors (torch.FloatTensor): Radiance at the given rays of shape [B, Nr, 3].
            weights (torch.FloatTensor): Weights of the given rays of shape [B, Nr, 1].

        """
        # Sample points along the rays

        tau = sigma * deltas
        ray_colors, ray_dapth, ray_alpha = exponential_integration(rgb, tau, depth, exclusive=True)

        return ray_colors, ray_dapth, ray_alpha


    def forward(self):
        """Forward pass of the network. 
        TODO: Adjust the neural rendering pipeline according to your needs.

        Returns:
            rgb (torch.FloatTensor): Ray codors of shape [B, Nr, 3].

        """
        B, Nr = self.ray_orig.shape[:2]

        # Step 1 : Sample points along the rays
        self.coords, self.z_vals, self.deltas = self.sample_points(
                                self.ray_orig, self.ray_dir, near=self.cfg.near, far=self.cfg.far,
                                num_points=self.cfg.num_pts_per_ray)

        # Step 2 : Predict radiance and volume density at the sampled points
        self.rgb, self.sigma = self.predict_radience(self.coords)

        # Step 3 : Volume rendering to compute the RGB color at the given rays
        self.ray_colors, self.ray_depth, self.ray_alpha = self.volume_render(self.rgb, self.sigma, self.z_vals, self.deltas)

        # Step 4 : Compositing with background color
        if self.cfg.bg_color == 'white':
            bg = torch.ones(B, Nr, 3, device=self.ray_colors.device)
            self.rgb = (1 - self.ray_alpha) * bg + self.ray_alpha * self.ray_colors
        else:
            self.rgb = self.ray_alpha * self.ray_colors


    def backward(self):
        """Backward pass of the network.
        TODO: You can also desgin your own loss function.
        """

        loss = 0.0
        rgb_loss = torch.abs(self.rgb -  self.img_gts).mean()

        loss = rgb_loss # + any other loss terms

        self.log_dict['rgb_loss'] += rgb_loss.item()
        self.log_dict['total_loss'] += loss.item()

        loss.backward()

    def step(self, data):
        """A signle training step.
        """

        # Get rays, and put them on the device
        self.ray_orig = data['rays'][..., :3].to(self.device)
        self.ray_dir = data['rays'][..., 3:].to(self.device)
        self.img_gts = data['imgs'].to(self.device)

        self.optimizer.zero_grad()
            
        self.forward()
        self.backward()
        
        self.optimizer.step()
        self.log_dict['total_iter_count'] += 1
        self.log_dict['image_count'] += self.ray_orig.shape[0]

    def render(self, ray_orig, ray_dir):
        """Render a full image for evaluation.
        """
        ray_orig = ray_orig.to(self.device)
        ray_dir = ray_dir.to(self.device)
        self.mlp.to(self.device)

        B, Nr = ray_orig.shape[:2]
        coords, depth, deltas = self.sample_points(ray_orig, ray_dir, near=self.cfg.near, far=self.cfg.far,
                                num_points=self.cfg.num_pts_per_ray_render)
        rgb, sigma = self.predict_radience(coords)
        ray_colors, ray_depth, ray_alpha= self.volume_render(rgb, sigma, depth, deltas)
        
        if self.cfg.bg_color == 'white':
            bg = torch.ones(B, Nr, 3, device=ray_colors.device)
            render_img = (1 - ray_alpha) * bg + ray_alpha * ray_colors
        else:
            render_img = ray_alpha * ray_colors

        return render_img, ray_depth, ray_alpha

    def reconstruct_3D(self, save_dir, epoch=0, sigma_threshold = 50., chunk_size=8192):
        """Reconstruct the 3D shape from the volume density.
        """

        self.mlp.to(self.device)

        # Mesh evaluation
        window_x = torch.linspace(-1., 1., steps=RES, device=self.device)
        window_y = torch.linspace(-1., 1., steps=RES, device=self.device)
        window_z = torch.linspace(-1., 1., steps=RES, device=self.device)
        
        coord = torch.stack(torch.meshgrid(window_x, window_y, window_z, indexing='ij')).permute(1, 2, 3, 0).reshape(-1, 3).contiguous()

        _points = torch.split(coord, int(chunk_size), dim=0)
        voxels = []
        for _p in _points:
            _, sigma = self.predict_radience(_p) 
            voxels.append(sigma)
        voxels = torch.cat(voxels, dim=0)

        np_sigma = torch.clip(voxels, 0.0).reshape(RES, RES, RES).cpu().numpy()

        vertices, faces = mcubes.marching_cubes(np_sigma, sigma_threshold)
        #vertices = ((vertices - 0.5) / (res/2)) - 1.0
        vertices = (vertices / (RES-1)) * 2.0 - 1.0

        h = trimesh.Trimesh(vertices=vertices, faces=faces)
        h.export(os.path.join(save_dir, '%04d.obj' % (epoch)))


    def log(self, step, epoch):
        """Log the training information.
        """
        log_text = 'STEP {} - EPOCH {}/{}'.format(step, epoch, self.cfg.epochs)
        self.log_dict['total_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'])
        self.log_dict['rgb_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | rgb loss: {:>.3E}'.format(self.log_dict['rgb_loss'])

        log.info(log_text)

        for key, value in self.log_dict.items():
            if 'loss' in key:
                wandb.log({key: value}, step=step)
        self.init_log_dict()

    def validate(self, loader, img_shape, step=0, epoch=0, sigma_threshold = 50., chunk_size=8192, save_img=False):
        """validation function for generating final results.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # To avoid CUDA out of memory
        self.eval()

        log.info("Beginning validation...")
        log.info(f"Loaded validation dataset with {len(loader)} images at resolution {img_shape[0]}x{img_shape[1]}")


        self.valid_mesh_dir = os.path.join(self.log_dir, "mesh")
        log.info(f"Saving reconstruction result to {self.valid_mesh_dir}")
        if not os.path.exists(self.valid_mesh_dir):
            os.makedirs(self.valid_mesh_dir)

        if save_img:
            self.valid_img_dir = os.path.join(self.log_dir, "img")
            log.info(f"Saving rendering result to {self.valid_img_dir}")
            if not os.path.exists(self.valid_img_dir):
                os.makedirs(self.valid_img_dir)

        psnr_total = 0.0

        wandb_img = []
        wandb_img_gt = []

        with torch.no_grad():
            # Evaluate 3D reconstruction
            self.reconstruct_3D(self.valid_mesh_dir, epoch=epoch,
                            sigma_threshold=sigma_threshold, chunk_size=chunk_size)

            # Evaluate 2D novel view rendering
            for i, data in enumerate(tqdm(loader)):
                rays = data['rays'].to(self.device)          # [1, Nr, 6]
                img_gt = data['imgs'].to(self.device)      # [1, Nr, 3]
                mask = data['masks'].repeat(1, 1, 3).to(self.device)

                _rays = torch.split(rays, int(chunk_size), dim=1)
                pixels = []
                for _r in _rays:
                    ray_orig = _r[..., :3]          # [1, chunk, 3]
                    ray_dir = _r[..., 3:]           # [1, chunk, 3]
                    ray_rgb, ray_depth, ray_alpha = self.render(ray_orig, ray_dir)
                    pixels.append(ray_rgb)

                pixels = torch.cat(pixels, dim=1)

                psnr_total += psnr(pixels, img_gt)

                img = (pixels).reshape(*img_shape, 3).cpu().numpy() * 255
                gt = (img_gt).reshape(*img_shape, 3).cpu().numpy() * 255
                wandb_img.append(wandb.Image(img))
                wandb_img_gt.append(wandb.Image(gt))

                if save_img:
                    Image.fromarray(gt.astype(np.uint8)).save(
                        os.path.join(self.valid_img_dir, "gt-{:04d}-{:03d}.png".format(epoch, i)) )
                    Image.fromarray(img.astype(np.uint8)).save(
                        os.path.join(self.valid_img_dir, "img-{:04d}-{:03d}.png".format(epoch, i)) )

        wandb.log({"Rendered Images": wandb_img}, step=step)
        wandb.log({"Ground-truth Images": wandb_img_gt}, step=step)
                
        psnr_total /= len(loader)

        log_text = 'EPOCH {}/{}'.format(epoch, self.cfg.epochs)
        log_text += ' {} | {:.2f}'.format(f"PSNR", psnr_total)

        wandb.log({'PSNR': psnr_total, 'Epoch': epoch}, step=step)
        log.info(log_text)
        self.train()

    def save_model(self, epoch):
        """Save the model checkpoint.
        """

        fname = os.path.join(self.log_dir, f'model-{epoch}.pth')
        log.info(f'Saving model checkpoint to: {fname}')
        torch.save(self.mlp, fname)

    