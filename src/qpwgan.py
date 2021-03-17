import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

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

    def critic_iteration(self, data_batch, gen_batch):
        assert data_batch.shape[0] == gen_batch.shape[0]
        batch_size = data_batch.shape[0]
        #print(data_batch.shape, gen_batch.shape)
        batch = torch.cat([data_batch, gen_batch], dim=0)
        full_cost = torch.norm(
            batch[:, None, :] - batch[None, ...], p=self.q, dim=-1)**self.p / self.p
        #cost = torch.norm(data_batch[:, None, :] - gen_batch[None, ...], p=self.q, dim=-1)**self.p / self.p
        critic_vals = self.critic(batch)
        c_transform_vals = self.get_c_transform(full_cost, critic_vals)
        full_xi_vals = full_cost - \
            critic_vals[:, None] - c_transform_vals[None, :]
        xy_xi_vals = full_cost[:batch_size, batch_size:] - \
            critic_vals[:batch_size] - c_transform_vals[batch_size:]

        penalty1 = self.xy_penalty(xy_xi_vals)
        penalty2 = self.admissable_penalty(full_xi_vals)

        critic_x = critic_vals[:batch_size].mean()
        c_transform_y = c_transform_vals[batch_size:].mean()

#         self.reg_coef1 = 1 / (1 * c_transform_vals[batch_size:].shape[0] ) ** 2
#         self.reg_coef2 = 1 / (2 * c_transform_vals[batch_size:].shape[0] ) ** 2

        loss = - critic_x - c_transform_y + penalty1 * \
            self.reg_coef1 + penalty2 * self.reg_coef2
        self.critic_optimizer.zero_grad()
        loss.backward(retain_graph=True)  # not sure about this
        self.critic_optimizer.step()

        return loss.item(), critic_x, c_transform_y

    def iteration(self, data_batch):
        batch_size = data_batch.shape[0]
        gen_batch = self.generator.sample(batch_size, self.device)

        for p in self.critic.parameters():
            p.requires_grad = True

        for _ in range(self.n_critic_iter):
            critic_loss, critic_x, c_transform_y = self.critic_iteration(
                data_batch, gen_batch)

        for p in self.critic.parameters():
            p.requires_grad = False

        batch = torch.cat([data_batch, gen_batch], dim=0)
        full_cost = torch.norm(
            batch[:, None, :] - batch[None, ...], p=self.q, dim=-1)**self.p / self.p
        critic_vals = self.critic(batch)
        c_transform_vals = self.get_c_transform(full_cost, critic_vals)
        critic_x = critic_vals[:batch_size].mean()
        c_transform_y = c_transform_vals[batch_size:].mean()

        gen_loss = critic_x + c_transform_y
        self.gen_optimizer.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()

        return critic_loss, gen_loss.item()

    def train(self, n_epoch=None):
        n_epoch = self.n_epoch if n_epoch is None else n_epoch
        gen_loss_history = []
        critic_loss_history = []
        for epoch in range(n_epoch):
            epoch_gen_loss = 0
            for data_batch in self.trainloader:
                data_batch = data_batch.to(self.device)
                critic_loss, gen_loss = self.iteration(data_batch)
                epoch_gen_loss += gen_loss / len(self.trainloader)
                gen_loss_history.append(gen_loss)
                critic_loss_history.append(critic_loss)
            if self.verbose:
                print(f'Epoch {epoch}, gen loss: {epoch_gen_loss:.3f}')

    def train_gaussian_mixture(self, target_sample, n_iter: int = 500):
        target_sample = target_sample.to(self.device)
        gen_loss_history = []
        critic_loss_history = []
        for iter_ in range(n_iter):
            critic_loss, gen_loss = self.iteration(target_sample)
            gen_loss_history.append(gen_loss)
            critic_loss_history.append(critic_loss)
            if iter_ % 10 == 0 and iter_ > 0:
                if self.verbose:
                    print(
                        f'Epoch {iter_}/{n_iter}, gen loss: {gen_loss:.3f}, critic_loss : {critic_loss:.3f}')
            if iter_ % 100 == 0 and iter_ > 0:
                fig = plt.figure()
                sample = self.generator.sample(
                    batch_size=100).to('cpu').detach().numpy()
                plt.scatter(sample[:, 0], sample[:, 1],
                            color='red', s=10, label='generated')
                target_sample = target_sample.detach().cpu().numpy()
                plt.scatter(target_sample[:,
                                          0],
                            target_sample[:,
                                          1],
                            color='cornflowerblue',
                            s=10,
                            label='original data')
                target_sample = torch.Tensor(target_sample).to(self.device)
                plt.title(f'p={self.p} iteration {iter_}')
                plt.plot([0], [0], marker='x', markersize=10, color='black')
                plt.legend()
                plt.savefig(
                    Path(
                        '../test',
                        f'iteration_{iter_}_sampled_gaussian_p={self.p}.png'))
                plt.close()

            if iter_ % 100 == 0 and iter_ > 0:
                fig = plt.figure()
                target_sample = target_sample.detach().cpu().numpy()
                plt.scatter(target_sample[:,
                                          0],
                            target_sample[:,
                                          1],
                            color='cornflowerblue',
                            s=10,
                            label='original data')
                target_sample = torch.Tensor(target_sample).to(self.device)

                delta = 0.02
                x = np.arange(-1.0, 1.0, delta)
                y = np.arange(-1.0, 1.0, delta)
                X, Y = np.meshgrid(x, y)
                z = np.zeros((X.shape[0] * X.shape[1], 2))
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        z[i * X.shape[0] + j] = np.append(X[i][j], Y[i][j])

                input = torch.Tensor(
                    X.shape[0] *
                    X.shape[1],
                    2).to(
                    self.device)
                output_real = self.critic(input).detach().cpu().numpy()
                output_real[np.isnan(output_real)] = 0

#                 CS = plt.pcolormesh(X, Y, output_real.reshape(X.shape), cmap=plt.cm.rainbow)
                CS = plt.contour(X, Y, output_real.reshape(
                    X.shape), levels=np.linspace(-1, 1, 10))
                CB = plt.colorbar(CS, shrink=0.9, extend='both')
                plt.title(f'Critic values p={self.p} iteration {iter_}')
                plt.legend()
                plt.savefig(
                    Path(
                        '../test',
                        f'iteration_{iter_}_critic_p={self.p}.png'))
                plt.close()

    @staticmethod
    def get_c_transform(cost, critic_vals):
        x = torch.min(cost - critic_vals, dim=0)[0]
        return x

    @staticmethod
    def xy_penalty(xy_xi_vals):
        return torch.mean(xy_xi_vals**2)

    @staticmethod
    def admissable_penalty(full_xi_vals):
        return torch.mean(torch.clamp(full_xi_vals, min=0)**2)
