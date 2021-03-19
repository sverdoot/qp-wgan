import torch
import wandb
from torch import nn
from torch.nn import functional as F

from matplotlib import pyplot as plt
from pathlib import Path


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
                m.weight.data.normal_(0.0, 0.05)
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
                m.weight.data.normal_(0.0, 0.05)
                m.bias.data.fill_(0)

    def forward(self, x):
        out = self.net(x).squeeze(1)
        return out


def mnist_callback(**kw):
    dump_dir = kw.get('dump_dir', '../test')
    inv_normalize = kw.get('inv_normalize', lambda x: x)

    def callback(wgan, epoch, *args, **kwargs):
        wgan.generator.eval()
        wgan.critic.eval()
        if epoch % 1 != 0:
            return
        sample = wgan.generator.sample(100, device=wgan.device)
        sample = sample.reshape(-1, 28, 28).detach().cpu()
        _, axs = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
        # plt.title(f'({wgan.q}, {wgan.p})-WGAN')
        for ax, im in zip(axs.flat, sample):
            im_ = inv_normalize(im[None])[0]
            ax.imshow(im_)
            ax.set_aspect('equal')
            ax.axis('off')
        plt.savefig(Path(dump_dir, f'{wgan.q}_{wgan.p}_mnist_{epoch}epoch.pdf'))
        plt.close()
        wandb.log({"examples": [wandb.Image(i) for i in sample]})

    return callback

        # data = train_dataset.data[:100]
        # _, axs = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
        # for ax, im in zip(axs.flat, data):
        #     ax.imshow(im)
        #     ax.set_aspect('equal')
        #     ax.axis('off')
        # plt.savefig(Path(dump_dir, f'mnist.pdf'))
        # plt.close()
