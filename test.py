import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime
import logging 
import numpy as np
from tqdm import tqdm
import torch
import torch.serialization
from PIL import Image

from torch.utils.data import DataLoader
from lib.datasets.multiview_dataset import MultiviewDataset
from lib.model.trainer import Trainer
from lib.model.mlps import MLP
from lib.model.positional_encoding import PositionalEncoding

from lib.utils.config import *

def main(config):
    
    test_dataset = MultiviewDataset(config.data_root,
                                     mip=0, 
                                     split='test') 
    test_loader = DataLoader(dataset=test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True)  
   

    img_shape = test_dataset.img_shape # Height and width of the rendered image

    pe = PositionalEncoding (config.num_freq, config.max_freq)


    torch.serialization.add_safe_globals([MLP])

    ckpt_path = os.path.join(config.pretrained_root, config.model_name)
    network = torch.load(ckpt_path, weights_only=False)

    trainer = Trainer(config, network, pe, config.pretrained_root)
    test_save_path = os.path.join(config.pretrained_root, 'test')
    if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)

    with torch.no_grad():
        
        trainer.reconstruct_3D(test_save_path, epoch=9999,
                sigma_threshold=config.sigma_thres, chunk_size=config.chunk_size)
                
        for i, data in enumerate(tqdm(test_loader)):
            rays = data['rays'].cuda()          # [1, Nr, 6]
            _rays = torch.split(rays, int(config.chunk_size), dim=1)
            pixels = []
            for _r in _rays:
                ray_orig = _r[..., :3]          # [1, chunk, 3]
                ray_dir = _r[..., 3:]           # [1, chunk, 3]
                ray_rgb, _, _= trainer.render(ray_orig, ray_dir)
                pixels.append(ray_rgb)

            pixels = torch.cat(pixels, dim=1)

            img = (pixels).reshape(*img_shape, 3).cpu().numpy() * 255

            Image.fromarray(img.astype(np.uint8)).save(
                os.path.join(test_save_path, "test-{:03d}.png".format(i)) )


if __name__ == "__main__":

    parser = parse_options()
    parser.add_argument('--pretrained-root', type=str, required=True, help='pretrained model path')
    parser.add_argument('--model-name', type=str, required=True, help='load model name')

    args, args_str = argparse_to_str(parser)
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s|%(levelname)8s| %(message)s',
                        handlers=handlers)
    logging.info(f'Info: \n{args_str}')
    main(args)