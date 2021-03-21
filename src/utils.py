import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import wandb
#import random
from metrics import closest_samples
from matplotlib import pyplot as plt

DUMP_DIR = 'dump'
DATA_DIR = 'data'
FIGS_DIR = 'figs'
CHECKS_DIR = 'checkpoints'


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, *self.shape)


def nearest_distance_callback(wgan, epoch, *args, **kwargs):
    if epoch == wgan.n_epoch - 1:

        dump_dir = kwargs.get('dump_dir', '../test')
        wgan.generator.eval()
        wgan.critic.eval()

        samples_from_dataset = []
        for i_id, i in enumerate(wgan.trainloader):
            if i_id > 100:  # for not waiting too long
                break
            samples_from_dataset.append(i[0])
        samples_from_dataset = torch.stack(samples_from_dataset)
        samples_from_dataset = samples_from_dataset.reshape(
            -1, samples_from_dataset.shape[2])
        sample = wgan.generator.sample(
            2000, device=wgan.device).detach().cpu()

        distances = closest_samples(samples_from_dataset, sample)

        json.dump(distances, Path(
            dump_dir,
            f'{wgan.task}_distances__{wgan.q}__{wgan.p}__critic_{wgan.n_critic_iter}__epoch_{epoch}.json').open('w')
        )

    return


def save_callback(wgan, epoch, *args, **kwargs):
    if epoch == wgan.n_epoch - 1:
        dump_dir = kwargs.get('dump_dir', '../models')
        torch.save({
            'epoch': epoch,
            'generator_state_dict': wgan.generator.state_dict(),
            'generator_optimizer_state_dict': wgan.gen_optimizer.state_dict(),
            'critic_state_dict': wgan.critic.state_dict(),
            'critic_optimizer_state_dict': wgan.critic_optimizer.state_dict(),
        }, Path(dump_dir, f'{wgan.task}_model__{wgan.q}__{wgan.p}__critic_{wgan.n_critic_iter}__epoch_{epoch}.pt'))

    return


def plotting_callback(wgan, epoch, *args, **kwargs):
    dump_dir = kwargs.get('dump_dir', '../test')
    inv_normalize = kwargs.get('inv_normalize', lambda x: x)

    task = wgan.task

    if task == 'cifar':
        height = width = 32
        n_channels = 3
    elif task == 'mnist':
        height = width = 28
        n_channels = 1
    else:
        raise NotImplementedError(
            f"Plotting for task {wgan.task} hasn't been implemented yet.")

    wgan.generator.eval()
    wgan.critic.eval()

    if epoch % 1 != 0:
        return

    sample = wgan.generator.sample(100, device=wgan.device)
    sample = sample.reshape(-1, n_channels, height, width).detach().cpu()
    _, axs = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
    plt.title(f'({wgan.q}, {wgan.p})-WGAN')
    for ax, im in zip(axs.flat, sample):
        im_ = inv_normalize(im)
        ax.imshow(im_.permute(1, 2, 0))
        ax.set_aspect('equal')
        ax.axis('off')
    plt.savefig(Path(
        dump_dir, f'{wgan.task}_{wgan.q}_{wgan.p}_{wgan.n_critic_iter}_{task}_{epoch}epoch.pdf'))
    plt.close()
    wandb.log({f"{task}_examples": [wandb.Image(i) for i in sample]})

    return
