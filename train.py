import os, sys
from datetime import datetime
import logging 
import numpy as np
import torch
import random
import shutil
import tempfile
import wandb

#from aitviewer.viewer import Viewer
#from aitviewer.renderables.point_clouds import PointClouds
#from aitviewer.renderables.arrows import Arrows
#from aitviewer.renderables.meshes import Meshes

from torch.utils.data import DataLoader
from lib.datasets.multiview_dataset import MultiviewDataset
from lib.model.trainer import Trainer
from lib.model.mlps import MLP, get_activation_class
from lib.model.positional_encoding import PositionalEncoding

from lib.utils.config import *

def create_archive(save_dir, config):

    with tempfile.TemporaryDirectory() as tmpdir:

        shutil.copy(config, os.path.join(tmpdir, 'config.yaml'))
        shutil.copy('train.py', os.path.join(tmpdir, 'train.py'))
        shutil.copy('test.py', os.path.join(tmpdir, 'test.py'))

        shutil.copytree(
            os.path.join('lib'),
            os.path.join(tmpdir, 'lib'),
            ignore=shutil.ignore_patterns('__pycache__'))

        shutil.make_archive(
            os.path.join(save_dir, 'code_copy'),
            'zip',
            tmpdir) 


def main(config):

    # Set random seed.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    log_dir = os.path.join(
            config.save_root,
            config.exp_name,
            f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )

    # Backup code.
    create_archive(log_dir, config.config)
    
    # Initialize dataset and dataloader.

    dataset = MultiviewDataset(config.data_root,
                               mip=0,         # We don't need to downsample the data for training
                               bg_color=config.bg_color, # Use white background for evaluation
                               sample_rays=True,         # Only sample a subset of rays for training
                               n_rays=config.num_rays_per_img)
    
    valid_dataset = MultiviewDataset(config.data_root,
                                     mip=2, 
                                     split='val')

    rander_shape = valid_dataset.img_shape # Height and width of the rendered image

    loader = DataLoader(dataset=dataset, 
                        batch_size=config.batch_size, 
                        shuffle=True, 
                        num_workers=config.workers,
                        pin_memory=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True)  

    # Initialize network and trainer.

    pe = PositionalEncoding (config.num_freq, config.max_freq)

    network = MLP(pe.out_dim, config.out_dim, activation=get_activation_class(config.activation), 
                  num_layers= config.num_layers , hidden_dim=config.hidden_dim,
                  skip=[config.skip] )

    trainer = Trainer(config, network, pe, log_dir)

    # Setup wandb for logging

    if config.wandb_id is not None:
        wandb_id = config.wandb_id
    else:
        wandb_id = wandb.util.generate_id()
        with open(os.path.join(log_dir, 'wandb_id.txt'), 'w+') as f:
            f.write(wandb_id)

    wandb_mode = "disabled" if (not config.wandb) else "online"
    wandb.init(id=wandb_id,
               project=config.wandb_name,
               config=config,
               name=os.path.basename(log_dir),
               resume="allow",
               settings=wandb.Settings(start_method="fork"),
               mode=wandb_mode,
               dir=log_dir)
    wandb.watch(network)

    # Main training loop

    global_step = 0
    for epoch in range(config.epochs):
        for data in loader:
            trainer.step(data)
            if global_step % config.log_every == 0:
                trainer.log(global_step, epoch)
            global_step += 1

        if epoch % config.save_every == 0:
            trainer.save_model(epoch)

        if epoch % config.valid_every == 0:
            trainer.validate(valid_loader, rander_shape, step=global_step, epoch=epoch, 
                             sigma_threshold=config.sigma_thres,
                             chunk_size=config.chunk_size, save_img=config.save_img)

    wandb.finish()

if __name__ == "__main__":

    parser = parse_options()
    args, args_str = argparse_to_str(parser)
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)
    logging.info(f'Info: \n{args_str}')
    main(args)