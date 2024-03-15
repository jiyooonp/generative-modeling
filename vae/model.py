import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        """
        TODO 2.1 : Fill in self.convs following the given architecture
         Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU()
                (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (3): ReLU()
                (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (5): ReLU()
                (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )
        """

        #TODO 2.1: fill in self.fc, such that output dimension is self.latent_dim
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )
        self.fc = nn.Linear(256*16, self.latent_dim) #(input_shape[1]/8)**2
        # self.fc = nn.Linear(256, self.latent_dim) #(input_shape[1]/8)**2
        self.ffc = nn.Linear(256*16, self.latent_dim) #(input_shape[1]/8)**2


    def forward(self, x):
        #TODO 2.1 : forward pass through the network, output should be of dimension : self.latent_dim
        import ipdb;
        # ipdb.set_trace()
        b, c, h, w = x.shape # torch.Size([256, 3, 32, 32])
        x = self.convs(x)   # torch.Size([256, 256, 4, 4])
        # ipdb.set_trace()
        x = x.view(x.shape[0], -1) # torch.Size([256, 4096])
        # ipdb.set_trace()
        x = self.ffc(x) # torch.Size([256, 256])
        # ipdb.set_trace()

        return x

class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        #TODO 2.4: fill in self.fc, such that output dimension is 2*self.latent_dim
        self.fc = nn.Linear(256 * 16, 2*self.latent_dim)
        # self.fc = nn.Linear(latent_dim, 2*self.latent_dim)
        # self.fc = nn.Sequential(
        #     nn.Linear(input_shape[1], 2 * self.latent_dim)
        # )
        self.enc_mu = torch.nn.Linear(2*self.latent_dim, self.latent_dim)
        self.enc_log_sigma = torch.nn.Linear(2*self.latent_dim, self.latent_dim)


    def forward(self, x):
        #TODO 2.4: forward pass through the network.
        # should return a tuple of 2 tensors, mu and log_std
        # x = super().forward(x)
        x = self.convs(x)
        # x = x.view(-1, x.shape[0])
        x = x.view(x.shape[0], -1)
        # import ipdb; ipdb.set_trace()
        x = self.fc(x)
        # mu, log_std = torch.chunk(x, 2, dim=1)
        # import ipdb; ipdb.set_trace()
        mu = self.enc_mu(x)
        log_std = self.enc_log_sigma(x)         # https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py line 23
        return mu, log_std


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        #TODO 2.1: fill in self.base_size
        self.base_size = output_shape[1] // 8

        """
        TODO 2.1 : Fill in self.deconvs following the given architecture
        Sequential(
                (0): ReLU()
                (1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (2): ReLU()
                (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (4): ReLU()
                (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (6): ReLU()
                (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        """
        self.fc = nn.Linear(self.latent_dim, 256 * (self.base_size ** 2))
        self.deconvs = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )


    def forward(self, z):
        #TODO 2.1: forward pass through the network, first through self.fc, then self.deconvs.
        out = self.fc(z)
        out = out.view(-1, 256, self.base_size, self.base_size)
        out = self.deconvs(out)
        return out

class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
    #NOTE: You don't need to implement a forward function for AEModel. For implementing the loss functions in train.py, call model.encoder and model.decoder directly.
