import torch
import wandb
from torch import nn
from torch.nn import functional as F

from matplotlib import pyplot as plt
from pathlib import Path
import json

from metrics import closest_samples


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
        plt.title(f'({wgan.q}, {wgan.p})-WGAN')
        for ax, im in zip(axs.flat, sample):
            im_ = inv_normalize(im[None])[0]
            ax.imshow(im_)
            ax.set_aspect('equal')
            ax.axis('off')
        plt.savefig(
            Path(dump_dir, f'{wgan.q}_{wgan.p}_{wgan.n_critic_iter}_mnist_{epoch}epoch.pdf'))
        plt.close()
        wandb.log({"mnist_examples": [wandb.Image(i) for i in sample]})
        
    def metric_callback(wgan, epoch, *args, **kwargs):
        if epoch == wgan.n_epoch - 1:
            dump_dir = kw.get('dump_dir', '../test')
            wgan.generator.eval()
            wgan.critic.eval()
            
            samples_from_dataset = []
            for i_id, i in enumerate(trainloader):
                if i_id > 50: # for not waiting too long
                    break
                samples_from_dataset.append(i[0])
            samples_from_dataset = torch.stack(samples_from_dataset)
            samples_from_dataset = samples_from_dataset.reshape(-1, samples_from_dataset.shape[2])
            sample = wgan.generator.sample(2000, device=wgan.device).detach().cpu()
            
            distances = closest_samples(samples, generated)
            
            json.dump(distances, Path(dump_dir, f'mnist_distances__{wgan.q}__{wgan.p}__critic_{wgan.n_critic_iter}__epoch_{epoch}.json').open('w'))
            
    def save_callback(wgan, epoch, *args, **kwargs):
        if epoch == wgan.n_epoch - 1:
            dump_dir = kw.get('dump_dir', '../models')
            torch.save({
                'epoch': epoch,
                'generator_state_dict': wgan.generator.state_dict(),
                'generator_optimizer_state_dict': wgan.gen_optimizer.state_dict(),
                'critic_state_dict': wgan.critic.state_dict(),
                'generator_optimizer_state_dict': wgan.critic_optimizer.state_dict(),
                'epoch': epoch
            }, Path(dump_dir, f'mnist_model__{wgan.q}__{wgan.p}__critic_{wgan.n_critic_iter}__epoch_{epoch}.pt'))
        
    callbacks = [callback, metric_callback, save_callback]
    return callbacks

    # data = train_dataset.data[:100]
    # _, axs = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
    # for ax, im in zip(axs.flat, data):
    #     ax.imshow(im)
    #     ax.set_aspect('equal')
    #     ax.axis('off')
    # plt.savefig(Path(dump_dir, f'mnist.pdf'))
    # plt.close()
