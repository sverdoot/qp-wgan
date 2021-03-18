import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 128
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(.2),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 28 * 28),
            nn.LeakyReLU(.2),
        )

    def init_weights(self):
        for m in self.net:
            if type(m) == nn.Linear:
                m.weight.data.normal_(0.0, 0.1)
                m.bias.data.fill_(0)

    def forward(self, z):
        x = self.net(z)
        return x

    def sample(self, batch_size, device=torch.device('cuda')):
        z = torch.randn(batch_size, self.latent_dim).to(device)
        x = self.forward(z)
        return x


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(.2),
            nn.Dropout(.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(.2),
            nn.Dropout(.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(.2),
            nn.Dropout(.3),
            nn.Linear(256, 1)
        )

    def init_weights(self):
        for m in self.net:
            if type(m) == nn.Linear:
                m.weight.data.normal_(0.0, 0.1)
                m.bias.data.fill_(0)

    def forward(self, x):
        #print(x)
        out = self.net(x).squeeze(1)  # .reshape(-1)
        return out
