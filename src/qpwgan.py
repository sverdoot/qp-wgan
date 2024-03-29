import torch
import wandb
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
from typing import List, Tuple

#from src.discrete_measures import compute_wasserstein
from discrete_measures import compute_wasserstein

torch.autograd.set_detect_anomaly(True)


class QPWGAN():
    def __init__(self, generator, critic, trainloader,
                 gen_optimizer, critic_optimizer, **kwargs):
        self.generator = generator
        self.critic = critic
        self.trainloader = trainloader
        self.gen_optimizer = gen_optimizer
        self.critic_optimizer = critic_optimizer
        self.n_epoch = kwargs.get('n_epoch', 30)
        self.n_critic_iter = kwargs.get('n_critic_iter', 5)
        self.q = kwargs.get('q', 2)
        self.p = kwargs.get('p', 1)
        self.verbose = kwargs.get('verbose', True)
        self.device = kwargs.get('device', 'cpu')
        self.reg_coef1 = kwargs.get('reg_coef1', 0.1)
        self.reg_coef2 = kwargs.get('reg_coef2', 0.1)
        self.search_space = kwargs.get('search_space', 'x')
        self.task = kwargs.get('task', 'unknown')

        assert self.search_space == 'x' or self.search_space == 'full'

    def critic_iteration(self, data_batch, gen_batch):
        assert data_batch.shape[0] == gen_batch.shape[0]
        batch_size = data_batch.shape[0]
        if self.search_space == 'full':
            batch = torch.cat([data_batch, gen_batch], dim=0)
            gen_batch_ = torch.cat([data_batch, gen_batch], dim=0)
        elif self.search_space == 'x':
            batch = data_batch
            gen_batch_ = gen_batch
        # full_cost = torch.norm(
        #     batch[:, None, :] - batch[None, ...], p=self.q, dim=-1)**self.p / self.p

        if batch.ndim == 2:
            full_cost = torch.norm(
                batch[:, None, :] - gen_batch_[None, ...], p=self.q, dim=-1)**self.p / self.p
        elif batch.ndim == 4:
            full_cost = torch.norm(
                batch[:, None, :, :, :] - gen_batch_[None, ...], p=self.q, dim=(2, 3, 4))**self.p / self.p

        #cost = torch.norm(data_batch[:, None, :] - gen_batch[None, ...], p=self.q, dim=-1)**self.p / self.p
        critic_vals = self.critic(batch)
        c_transform_vals = self.get_c_transform(full_cost, critic_vals)
        #print(full_cost.shape, critic_vals.shape, c_transform_vals.shape)
        full_xi_vals = full_cost - \
            critic_vals[:, None] - c_transform_vals[None, :]

        penalty2 = self.admissable_penalty(full_xi_vals)
        if self.search_space == 'full':
            xy_xi_vals = full_cost[:batch_size, batch_size:] - \
                critic_vals[:batch_size, None] - \
                c_transform_vals[None, batch_size:]
            c_transform_y = c_transform_vals[batch_size:].mean()
        elif self.search_space == 'x':
            xy_xi_vals = full_cost[:batch_size, :] - \
                critic_vals[:batch_size, None] - c_transform_vals[None, :]
            c_transform_y = c_transform_vals.mean()
        penalty1 = self.xy_penalty(xy_xi_vals)
        critic_x = critic_vals[:batch_size].mean()

        loss = - critic_x - c_transform_y + penalty1 * \
            self.reg_coef1 + penalty2 * self.reg_coef2
        self.critic_optimizer.zero_grad()
        loss.backward(retain_graph=True)  # not sure about this
        self.critic_optimizer.step()

        return loss.item(), critic_x, c_transform_y

    def iteration(self, data_batch, return_sample=False):
        batch_size = data_batch.shape[0]
        gen_batch = self.generator.sample(batch_size, self.device)

        for p in self.critic.parameters():
            p.requires_grad = True

        for _ in range(self.n_critic_iter):
            critic_loss, critic_x, c_transform_y = self.critic_iteration(
                data_batch, gen_batch)

        for p in self.critic.parameters():
            p.requires_grad = False

        #gen_batch = self.generator.sample(batch_size, self.device)

        if self.search_space == 'full':
            batch = torch.cat([data_batch, gen_batch], dim=0)
            gen_batch_ = torch.cat([data_batch, gen_batch], dim=0)
        elif self.search_space == 'x':
            batch = data_batch
            gen_batch_ = gen_batch
        if batch.ndim == 2:
            full_cost = torch.norm(
                batch[:, None, :] - gen_batch_[None, ...], p=self.q, dim=-1)**self.p / self.p
        elif batch.ndim == 4:
            full_cost = torch.norm(
                batch[:, None, :, :, :] - gen_batch_[None, ...], p=self.q, dim=(2, 3, 4))**self.p / self.p
        critic_vals = self.critic(batch)
        c_transform_vals = self.get_c_transform(full_cost, critic_vals)

        critic_x = critic_vals[:batch_size].mean()
        if self.search_space == 'full':
            c_transform_y = c_transform_vals[batch_size:].mean()
        elif self.search_space == 'x':
            c_transform_y = c_transform_vals.mean()

        gen_loss = critic_x + c_transform_y
        self.gen_optimizer.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()

        if return_sample:
            return critic_loss, gen_loss.item(), gen_batch.detach()

        return critic_loss, gen_loss.item()

    def train(self, n_epoch: int = None, callbacks=None):
        self.generator.train()
        self.critic.train()

        n_epoch = self.n_epoch if n_epoch is None else n_epoch
        if callbacks is not None and not isinstance(callbacks, (list, tuple)):
            callbacks = (callbacks,)
        gen_loss_history = []
        critic_loss_history = []
        if self.verbose:
            print(f'Length of dataloader: {len(self.trainloader)}')

        for epoch in range(n_epoch):
            epoch_gen_loss = 0
            epoch_critic_loss = 0
            for it, data_batch in enumerate(self.trainloader):
                if isinstance(data_batch, (list, tuple)):
                    data_batch = data_batch[0]
                data_batch = data_batch.to(self.device)
                critic_loss, gen_loss = self.iteration(data_batch)
                epoch_gen_loss += gen_loss / len(self.trainloader)
                epoch_critic_loss += critic_loss / len(self.trainloader)
                gen_loss_history.append(gen_loss)
                critic_loss_history.append(critic_loss)
                if self.verbose and it % 50 == 0:
                    print(
                        f'Iteration {it}, gen loss: {gen_loss:.3f}, critic loss: {critic_loss:.3f}')

            if self.verbose:
                print(f'Epoch {epoch}, gen loss: {epoch_gen_loss:.3f}')
                wandb.log({'gen loss': epoch_gen_loss,
                           'critic loss': epoch_critic_loss})

            if callbacks is not None:
                for callback in callbacks:
                    callback(self, epoch)

    def train_gaussian_mixture(self, n_epoch: int = 500, scaler=None):
        # TODO: substitute this method with existing 'train' using callbacks
        target_sample = self.trainloader.dataset.data
        target_sample_np = target_sample.detach().cpu().numpy()
        #target_sample_np = scaler.inverse_transform(target_sample.detach().cpu().numpy())


        gen_loss_history = []
        critic_loss_history = []
        wass_history = []
        for iter_ in range(n_epoch):
            epoch_gen_loss = 0
            epoch_critic_loss = 0
            epoch_w_loss = 0
            for data_batch in self.trainloader:
                if isinstance(data_batch, (list, tuple)):
                    data_batch = data_batch[0]
                data_batch = data_batch.to(self.device)
                critic_loss, gen_loss, gen_batch = self.iteration(
                    data_batch, return_sample=True)
                epoch_gen_loss += gen_loss / len(self.trainloader)
                epoch_critic_loss += critic_loss / len(self.trainloader)

                W, _ = compute_wasserstein(
                    target_sample, gen_batch, q=self.q, p=self.p)
                epoch_w_loss += W.item()**self.p / len(self.trainloader)
            wass_history.append(epoch_w_loss)
            gen_loss_history.append(gen_loss)
            critic_loss_history.append(critic_loss)
            # TODO: bootstrapping (or smth) for wasserstein metric estimation

            if iter_ % 10 == 0 and iter_ > 0:
                if self.verbose:
                    print(
                        f'Epoch {iter_}/{n_epoch}, gen loss: {gen_loss:.3f}, critic_loss : {critic_loss:.3f}')
            if iter_ % 100 == 0 and iter_ > 0:
                _ = plt.figure()
                sample = self.generator.sample(
                    batch_size=100, device=self.device).to('cpu').detach().numpy()
                #sample = scaler.inverse_transform(sample)
                kernel = stats.gaussian_kde(
                    np.unique(sample, axis=0).T)  # reshape(2, -1))

                delta = 0.02
                x = np.arange(-1.2, 1.2, delta)
                y = np.arange(-1.2, 1.2, delta)
                X, Y = np.meshgrid(x, y)
                input = np.zeros((
                    X.shape[0] *
                    X.shape[1],
                    2))
                input[:, 0] = X.reshape(-1)
                input[:, 1] = Y.reshape(-1)

                values = kernel.pdf(input.T).T
                CS = plt.contour(X, Y, values.reshape(
                    X.shape), levels=np.linspace(np.quantile(values, 0.75), 1, 10))

                plt.scatter(sample[:100, 0], sample[:100, 1],
                            color='red', s=10, label='generated')
                plt.scatter(target_sample_np[:,
                                          0],
                            target_sample_np[:,
                                          1],
                            color='cornflowerblue',
                            s=10,
                            label='original data', alpha=0.7)
                plt.title(f'p={self.p} iteration {iter_}')
                plt.plot([0], [0], marker='x', markersize=10, color='black')
                plt.legend()
                plt.savefig(
                    Path(
                        'dump',
                        f'gaussians_{iter_}_generator_p={self.p}.pdf'))
                plt.close()

                _ = plt.figure()
                plt.scatter(target_sample_np[:,
                                          0],
                            target_sample_np[:,
                                          1],
                            color='cornflowerblue',
                            s=10,
                            label='original data', alpha=0.7)

                input = torch.FloatTensor(input).to(self.device)
                output_real = self.critic(input).detach().cpu().numpy()
                output_real[np.isnan(output_real)] = 0

#                 CS = plt.pcolormesh(X, Y, output_real.reshape(X.shape), cmap=plt.cm.rainbow)
                CS = plt.contour(X, Y, output_real.reshape(
                    X.shape), levels=np.linspace(np.quantile(output_real, 0.75), output_real.max(), 10))

                CB = plt.colorbar(CS, shrink=0.9)  # , extend='both')
                plt.title(f'Critic values p={self.p} iteration {iter_}')
                plt.legend()
                plt.savefig(
                    Path(
                        'dump',
                        f'gaussians_{iter_}_critic_p={self.p}.pdf'))
                plt.close()

        return gen_loss_history, wass_history

    @staticmethod
    def get_c_transform(cost, critic_vals):
        x = torch.min(cost - critic_vals[:, None], dim=0)[0]
        return x

    @staticmethod
    def xy_penalty(xy_xi_vals):
        return torch.mean(xy_xi_vals**2)

    @staticmethod
    def admissable_penalty(full_xi_vals):
        return torch.mean(torch.clamp(full_xi_vals, max=0)**2)
