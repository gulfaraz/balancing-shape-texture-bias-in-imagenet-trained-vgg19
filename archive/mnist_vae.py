import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

from PIL import Image

class VAE(nn.Module):
    def __init__(self, n_channels=1, z_dim=20, n_class=10):
        super(VAE, self).__init__()
        self.z_size = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3)
        )

        feature_size = 18432

        self.encoder_to_latent = nn.Sequential(
            nn.Linear(feature_size, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.z_size),
            nn.ReLU(inplace=True)
        )

        self.latent_to_mu = nn.Linear(self.z_size, self.z_size)
        self.latent_to_logvar = nn.Linear(self.z_size, self.z_size)

        self.latent_to_decoder = nn.Sequential(
            nn.Linear(self.z_size, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, feature_size),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_channels, kernel_size=3),
            nn.Sigmoid()
        )

        self.classifier = nn.Linear(z_dim, n_class)
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.Tensor(torch.randn(*mu.size()))
        z = mu + std * esp
        return z
    
    def encode(self, x):
        h = self.encoder(x)
        features_shape = h.shape
        h = h.view(h.size(0), -1)
        z = self.encoder_to_latent(h)
        return z, features_shape
    
    def decode(self, z, features_shape=None):
        h = self.latent_to_decoder(z)
        h = h.view(*features_shape)
        x = self.decoder(h)
        return x
    
    def forward(self, x, classify=True):
        z, features_shape = self.encode(x)
        mu = self.latent_to_mu(z)
        logvar = self.latent_to_logvar(z)
        z = self.reparameterize(mu, logvar)
        if classify:
            return self.classifier(z)
        else:
            return self.decode(z, features_shape=features_shape), mu, logvar
    
    def set_mode(self, mode):
        for params in self.encoder.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        for params in self.encoder_to_latent.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        for params in self.latent_to_mu.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        for params in self.latent_to_logvar.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        for params in self.latent_to_decoder.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        for params in self.decoder.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        for params in self.classifier.parameters():
            params.requires_grad = (mode == 'train-classifier')


class VAE_(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20, n_class=10):
        super(VAE_, self).__init__()
        self.z_size = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, z_dim*2)
        )

        self.encoder_to_latent = lambda x: x

        self.latent_to_decoder = lambda x: x
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, image_size),
            nn.Sigmoid()
        )

        self.classifier = nn.Linear(z_dim, n_class)
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.Tensor(torch.randn(*mu.size()))
        z = mu + std * esp
        return z
    
    def forward(self, x, classify=True):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        if classify:
            return self.classifier(z)
        else:
            return self.decoder(z), mu, logvar
    
    def set_mode(self, mode):
        for params in self.encoder.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        # for params in self.encoder_to_latent.parameters():
        #     params.requires_grad = (mode == 'train-autoencoder')
        # for params in self.latent_to_mu.parameters():
        #     params.requires_grad = (mode == 'train-autoencoder')
        # for params in self.latent_to_logvar.parameters():
        #     params.requires_grad = (mode == 'train-autoencoder')
        # for params in self.latent_to_decoder.parameters():
        #     params.requires_grad = (mode == 'train-autoencoder')
        for params in self.decoder.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        for params in self.classifier.parameters():
            params.requires_grad = (mode == 'train-classifier')


def create_mnist_variational_autoencoder():
    return VAE(n_class=10)
    # return VAE(image_size=4096, h_dim=1024, z_dim=256, n_class=200)
