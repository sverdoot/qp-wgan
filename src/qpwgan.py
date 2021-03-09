import torch
from torch import nn
from torch.nn import functional as F

torch.autograd.set_detect_anomaly(True)


class QPWGAN():
    def __init__(self, generator, critic, trainloader, gen_optimizer, critic_optimizer, **kwargs):
        self.generator = generator
        self.critic = critic
        self.trainloader = trainloader
        self.gen_optimizer = gen_optimizer
        self.critic_optimizer = critic_optimizer
        self.n_epoch = kwargs.get('n_epoch')
        self.n_critic_iter = kwargs.get('n_critic_iter')
        self.q = kwargs.get('q')
        self.p = kwargs.get('p')
        self.verbose = kwargs.get('verbose', True)
        self.device = kwargs.get('device', 'cpu')
        
    def critic_iteration(self, data_batch, gen_batch):
        assert data_batch.shape[0] == gen_batch.shape[0]
        batch_size = data_batch.shape[0]
        #print(data_batch.shape, gen_batch.shape)
        batch = torch.cat([data_batch, gen_batch], dim=0)
        full_cost = torch.norm(batch[:, None, :] - batch[None, ...], p=self.q, dim=-1)**self.p / self.p
        #cost = torch.norm(data_batch[:, None, :] - gen_batch[None, ...], p=self.q, dim=-1)**self.p / self.p
        critic_vals = self.critic(batch)
        c_transform_vals = self.get_c_transform(full_cost, critic_vals)
        full_xi_vals = full_cost - critic_vals[:, None] - c_transform_vals[None, :]
        xy_xi_vals = full_cost[:batch_size, batch_size:] - critic_vals[:batch_size] - c_transform_vals[batch_size:]

        penalty1 = self.xy_penalty(xy_xi_vals)
        penalty2 = self.admissable_penalty(full_xi_vals)

        critic_x = critic_vals[:batch_size].mean()
        c_transform_y = c_transform_vals[batch_size:].mean()

        loss = - critic_x - c_transform_y + penalty1 + penalty2
        self.critic_optimizer.zero_grad()
        loss.backward(retain_graph=True) # not sure about this
        self.critic_optimizer.step()

        return loss.item(), critic_x, c_transform_y

    def iteration(self, data_batch):
        batch_size = data_batch.shape[0]
        gen_batch = self.generator.sample(batch_size)

        for p in self.critic.parameters():
                p.requires_grad = True

        for _ in range(self.n_critic_iter):
            critic_loss, critic_x, c_transform_y = self.critic_iteration(data_batch, gen_batch)

        for p in self.critic.parameters():
                p.requires_grad = False

        batch = torch.cat([data_batch, gen_batch], dim=0)
        full_cost = torch.norm(batch[:, None, :] - batch[None, ...], p=self.q, dim=-1)**self.p / self.p
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
        
    @staticmethod
    def get_c_transform(cost, critic_vals):
        return torch.min(cost - critic_vals[:, None], dim=0)[0]

    @staticmethod
    def xy_penalty(xy_xi_vals):
        return torch.mean(xy_xi_vals**2)

    @staticmethod
    def admissable_penalty(full_xi_vals):
        return torch.mean(torch.clamp(full_xi_vals, min=0)**2)
