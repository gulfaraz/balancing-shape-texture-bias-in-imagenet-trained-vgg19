import os
import torch
import torch.nn as nn
import torch.nn.init as init

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = torch.Tensor(std.data.new(std.size()).normal_())
    return mu + std*eps

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 64, 64
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B, 32, 64, 64
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 128, 128
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class BetaVAE_H_CLASSIFIER(BetaVAE_H):
    def __init__(self, vae_checkpoint_path, z_dim=10, nc=3, device='cuda:0'):
        super(BetaVAE_H_CLASSIFIER, self).__init__(z_dim=z_dim, nc=nc)

        if os.path.isfile(vae_checkpoint_path):
            checkpoint = torch.load(vae_checkpoint_path, map_location=device)
            self.load_state_dict(checkpoint['weights'])
        else:
            raise ValueError('Checkpoint Not Found: {}'.format(vae_checkpoint_path))

        self.classifier = nn.Linear(z_dim, 200)
    
    def forward(self, x):
        with torch.no_grad():
            encoded_x = self._encode(x)
            encoded_x = encoded_x[:, :self.z_dim] # take the mean
        return self.classifier(encoded_x)


def create_betavae(z_dim):
    def initialize_betavae():
        return BetaVAE_H(z_dim, 3)
    return initialize_betavae

def create_betavae_classifier(vae_checkpoint_path, z_dim, device):
    def initialize_betavae_classifier():
        return BetaVAE_H_CLASSIFIER(vae_checkpoint_path, z_dim=z_dim, nc=3, device=device)
    return initialize_betavae_classifier
