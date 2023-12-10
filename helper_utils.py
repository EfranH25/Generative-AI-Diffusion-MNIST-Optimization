import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
#torch.set_default_device('cuda')

def plot_images(images):
    """Used to plot images generated from diffusion.

    Args:
        images (Tesnor): Tesnor of images to plot
    """
    
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
    
def save_images(images, path, sample_image_name, epoch, **kwargs):
    """Used to save images generated from diffusion.

    Args:
        images (Tensor): Tesnor of images to save
        path (str): file path of where to save img
    """
    epoch = str(epoch)
    
    if os.path.exists(os.path.join(path, epoch)) == False:
        os.makedirs(os.path.join(path, epoch))
    
    #? saves images individually to path folder
    count = 0
    for i in images:
        image_name = os.path.join(path, epoch, f'{epoch}_{count}_img.jpg')
        img = torchvision.utils.make_grid(i.float(), **kwargs)
        torchvision.utils.save_image(img, image_name, normalize=True)
        count += 1
    print('saved images to epoch folder')
    
    #? saves images as a large grid
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(os.path.join(path, sample_image_name))
    print('saved images to epoch grid image')
    
    
def get_data(args):
    """Gets data and loads with Data Loader. Takes in arg parser fields

    Args:
        args (arg parser): _description_

    Returns:
        DataLoader: Dataloader of specified data used for diffusion training
    """
    channels_normalize = [.5 for x in range(args.channels)]
    
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(channels_normalize, channels_normalize)
    ])
    
    #? retrieves data
    if  args.dataset_path == 'mnist':        
        print('using mnist dataset')
        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
        print(dataset.data.shape)
        print('loaded mnist training dataset')
        #mnist_testset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
        
    elif args.dataset_path == 'cifar10':
        print('using cifar10 dataset')
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
        print(dataset.data.shape)
        print('loaded cifar10 training dataset')
        #cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
    else:
        print('using dataset in folder')
        dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
        print('loaded dataset in folder')
    
    #? loads data to dataloader
    print("Size of his data set: ", len(dataset))
    #dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator(device=args['device']))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def get_data_dics(args):
    """Gets data and loads with Data Loader. Takes in dicitonary fields as arguements

    Args:
        args (arg parser): _description_

    Returns:
        DataLoader: Dataloader of specified data used for diffusion training
    """
    
    channels_normalize = [.5 for x in range(args['channels'])]
    
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args['image_size'], scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(channels_normalize, channels_normalize)
    ])
    
    #? retrieves data
    if  args['dataset_path'] == 'mnist':        
        print('using mnist dataset')
        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
        print('loaded mnist training dataset')
        #mnist_testset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
    elif args['dataset_path'] == 'cifar10':
        print('using cifar10 dataset')
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
        print('loaded cifar10 training dataset')
        #cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
    else:
        print('using dataset in folder')
        dataset = torchvision.datasets.ImageFolder(args['dataset_path'], transform=transforms)
        print('loaded dataset in folder')
    
    #? loads data to dataloader
    print("Size of his data set: ", len(dataset))
    #dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, generator=torch.Generator(device=args['device']))
    dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    #args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"
    args.dataset_path = r'F:/Classes/COMPSCI 682/denoising-diffusion-pytorch-main/data/cifar-10/train/'
    #args.dataset_path = r"mnist"
    args.device = "cuda"
    args.lr = 3e-4
    
    results = get_data(args)
    print(type(get_data))
    

if __name__ == '__main__':
    launch()