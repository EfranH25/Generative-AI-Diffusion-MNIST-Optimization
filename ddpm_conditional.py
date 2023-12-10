
#? resources
#? vid 1: https://www.youtube.com/watch?v=HoKDTa5jHvg&t=177s
# explains how ddpm works
#? vid2: https://www.youtube.com/watch?v=TBCRlnwJtZU
# goes over implementation to ddpm

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from helper_utils import *

from modules import UNet, UNet_conditional, EMA
import logging 
from torch.utils.tensorboard import SummaryWriter
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

# library for quantization
try:
    import bitsandbytes as bnb
    print('imported bitsandbytes')
    
except:
    print('cant import bitsandbytes')
    bnb = None

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda", USE_GPU = True):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size #resoultion of image #: side note from video --> for higher resolutions, training seperate upsamplers instead of training on bigger resolution images
        self.device = device
        
        if USE_GPU and torch.cuda.is_available():
            print("CUDAAAAAAAAAAA")
            self.use_cuda = torch.device('cuda')
        else:
            self.use_cuda = torch.device('cpu')
        
        #? right now using simple beta schedule --> open AI using cosine scheduler        
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        #! try implementing cosine scheduler
        
    def prepare_noise_schedule(self):
        #? Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        """Adds noise to image. You can iteratively add noise to image but vid 1 showed 
        a simplification that adds noise in 1 step. Which is this implementation
        Args:
            x (_type_): _description_
            t (_type_): _description_

        Returns:
            _type_: returns image with noise added on
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        E = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * E, E
    
    def sample_timesteps(self, n):
        """_summary_

        Args:
            n (_type_): _description_

        Returns:
            _type_: _description_
        """
        #? needed for algorithm for training
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    

    def sample(self, model, n, labels, channels=3, cfg_scale=3):
        """implements algorithm 2 from the ddpm paper in vid 1

        Args:
            model (_type_): _description_
            n (int): number of images we want to sample 

        Returns:
            _type_: _description_
        """
        logging.info(f"Sampling {n} new images....")
        #? see here for why we set model.eval() https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
        #? essentially disables some some parts of torch for specific steps
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            #? create initial images by sampling over normal dist (step 1)
            x = torch.randn((n, channels, self.img_size, self.img_size)).to(self.device)
            
            #? step 2, 3, 4
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device) #? tensor of timestep
                predicted_noise = model(x, t, labels) #? feed that into model w/ current images
                
                #? noise
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                #? only want noise for timestemps greater than 1. done so b/c in last iteration, would make final outcome worse due to adding noise to finalized pixels
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                #? alter image by removed a little bit of noise
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        #? switch back to train    
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2 #? brings back value to 0-1 range 
        x = (x * 255).type(torch.uint8) #? bring back values to pixel range for viewing image
        return x

def train(args):
    setup_logging(args.run_name)
    device = args.device
    
    if not args.optim:
        print('using no optimization methods')
    
    if 'deep' in args.optim:
        print('using depthwise convolution optmization')
        groups = args.groups
    else:
        print('using depthwise convolution optmization')
        groups = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    model = UNet_conditional(c_in=args.channels, c_out=args.channels, size=args.image_size, time_dim=args.image_size*4, num_classes=args.num_classes, device=args.device, args=args, groups=groups).to(device)
    
    if 'quant' in args.optim:
        print('using quantization optimizer')
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    print('initialized model!')
    
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    dataloader = get_data(args)
    l = len(dataloader)
    print(f"Size of his data set after applying {args.batch_size} data points per batch:", len(dataloader))

    l = len(dataloader)
    

    print('starting loop')
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        #? alorithm 1 from vid 1  
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
        #for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            if np.random.random() < 0.1:
                labels = None
                
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print('generating sample images')
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), channels=args.channels, labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), channels=args.channels, labels=labels)

            print('saving sample images & models')
            sample_images_path = os.path.join("results", args.run_name)
            sample_image_name = f"{epoch}_.jpg"

            save_images(sampled_images, sample_images_path, sample_image_name, epoch=epoch)
            save_images(ema_sampled_images, sample_images_path, 'ema' + sample_image_name, epoch=epoch)

            torch.save(model, os.path.join("models", args.run_name, f"{epoch}_ckpt.pt"))
            torch.save(ema_model, os.path.join("models", args.run_name, f"{epoch}_ema_ckpt.pt"))
            torch.save(optimizer, os.path.join("models", args.run_name, f"{epoch}_optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args=[])
    args.run_name = "ddpm_conditional_ema_mnist"
    args.epochs = 300
    args.epochs = 5
    args.batch_size = 10
    
    args.dataset_path = r"cifar10"
    args.image_size = 64
    args.num_classes = 10
    args.channels = 3
    
    args.dataset_path = r"mnist"
    args.image_size = 56
    args.num_classes = 10
    args.channels = 1
    
    args.optim = ['quant', 'prune', 'deep']
    args.optim = ['deep']
    #args.optim = []
    args.groups= [1, 2, 2, 2, 2, 2, 2, 2, 1]
    # [1, 2, 2, 2, 2, 2, 2, 2, 1]
    # [1, 4, 4, 4, 4, 4, 4, 4, 1]
    # [1,32,64,128,128,64,32,1]
    
    args.device = "cuda"
    args.lr = 3e-4
    train(args)
    print('done')

    
if __name__ == '__main__':
    launch()