# TODO: check dimensions

from torch import nn
from torch.nn import functional as F

conv_kernel_size = 3
conv_stride = 2
conv_padding = 1

deconv_kernel_size = 2
deconv_stride = 2
deconv_padding = 0


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, conv_kernel_size, conv_stride, conv_padding)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(128, 2*128, conv_kernel_size, conv_stride, conv_padding)
        self.act2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(2*128, 4*128, conv_kernel_size, conv_stride, conv_padding)
        self.act3 = nn.LeakyReLU(0.2)

        self.linear = nn.Linear(4*4*4*128, 1)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        return self.linear(x)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.linear = nn.Linear(128, 4*4*4*128)
        self.deconv1 = nn.ConvTranspose2d(4*128, 2*128, deconv_kernel_size, deconv_stride, deconv_padding)
        self.bn1 = nn.BatchNorm2d(2*128)
        self.deconv2 = nn.ConvTranspose2d(2*128, 128, deconv_kernel_size, deconv_stride, deconv_padding)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 3, deconv_kernel_size, deconv_stride, deconv_padding)

    def forward(self, x):
        x = F.relu(self.linear(x)).reshape(-1, 4*128, 4, 4)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.tanh(self.deconv3(x))
        return x
