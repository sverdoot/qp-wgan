import torch
from torch import nn
from src.utils import Reshape


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
