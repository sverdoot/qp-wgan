import torch
import wandb
from torch import nn
from torch.nn import functional as F
from src.utils import Reshape
import sys

import matplotlib.pyplot as plt
from pathlib import Path


class Generator(nn.Module):
    '''
    Implementation of Generator architecture used in 
    the paper https://arxiv.org/pdf/1902.03642.pdf 
    For mode details see supplementary material.
    '''

    def __init__(self):  # , batch_size):
        super(Generator, self).__init__()

        # self.batch_size = batch_size
        self.latent_dim = 128
        self.net = nn.Sequential(
            # [batch_size, 128] -> [batch_size, 8192]
            nn.Linear(128, 4*4*4*128),
            nn.ReLU(),
            Reshape(4*128, 4, 4),
            nn.ConvTranspose2d(4*128, 2*128, 2, 2, 0),
            nn.BatchNorm2d(2*128),
            nn.ReLU(),
            nn.ConvTranspose2d(2*128, 128, 2, 2, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 2, 2, 0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.net(x)
        return x

    def sample(self, batch_size, device=torch.device('cuda')):
        z = torch.randn(batch_size, self.latent_dim).to(device)
        x = self.forward(z)
        return x


class Critic(nn.Module):
    '''

    '''

    def __init__(self):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(.2),
            nn.Conv2d(128, 2*128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(.2),
            nn.Conv2d(2*128, 4*128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(.2),
            nn.Flatten(),
            nn.Linear(4*4*4*128, 1)
        )

    def forward(self, x):
        out = self.net(x).squeeze(1)
        return out


def cifar_callback(**kw):
    dump_dir = kw.get('dump_dir', '../test')
    inv_normalize = kw.get('inv_normalize', lambda x: x)

    def callback(wgan, epoch, *fargs, **fkwargs):
        if epoch % 1 != 0:
            return
        sample = wgan.generator.sample(20, device=wgan.device)
        sample = sample.reshape(-1, 3, 32, 32).detach().cpu()
        _, axs = plt.subplots(nrows=4, ncols=5, figsize=(10, 15))
        for ax, im in zip(axs.flat, sample):
            im_ = inv_normalize(im)
            ax.imshow(torch.movedim(im_, 0, 2))
            ax.set_aspect('equal')
            ax.axis('off')
        plt.savefig(
            Path(dump_dir, f'{wgan.q}_{wgan.p}_{wgan.n_critic_iter}_cifar10_{epoch}epoch.pdf'))
        plt.close()
        wandb.log({"examples": [wandb.Image(i) for i in sample]})

    return callback
