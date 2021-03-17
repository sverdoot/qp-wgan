import torch
from torch import nn
from torch.nn import functional as F
from typing import List


class Generator(nn.Module):
    def __init__(self,
                 layers_dim: List[int] = [128, 128, 128, 2],
                 non_linear: nn.Module = nn.LeakyReLU(0.2),
                 output_non_linear: nn.Module = nn.LeakyReLU(0.2)):
        super(Generator, self).__init__()

        self.noise_dim = layers_dim[0]
        self.layers_dim = layers_dim
        layers_list = list()
        for i in range(len(layers_dim) - 1):
            layers_list.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
            if i != len(layers_dim) - 2:
                layers_list.append(
                    output_non_linear if i == (
                        len(layers_dim) -
                        2) else non_linear)
        layers_list[-3] = nn.Tanh()
        self.layer_seq = nn.Sequential(*layers_list)

    def forward(self, z):
        x = self.layer_seq(z)
        return x

    def sample(self, batch_size, device=torch.device('cuda')):
        z = ((torch.rand(batch_size, self.noise_dim) - 0.5) * 2).to(device)
        x = self.forward(z)
        return x


class Critic(nn.Module):
    def __init__(self,
                 layers_dim: List[int] = [2, 128, 128, 1],
                 non_linear: nn.Module = nn.LeakyReLU(0.2),
                 output_non_linear: nn.Module = nn.LeakyReLU(0.2)):
        super(Critic, self).__init__()

        self.input_dim = layers_dim[0]
        self.layers_dim = layers_dim
        layers_list = list()
        for i in range(len(layers_dim) - 1):
            layers_list.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
            if i != len(layers_dim) - 2:
                layers_list.append(
                    output_non_linear if i == (
                        len(layers_dim) -
                        2) else non_linear)
        layers_list[-3] = nn.Tanh()
        self.layer_seq = nn.Sequential(*layers_list)

    def forward(self, x):
        out = self.layer_seq(x)
        return out
