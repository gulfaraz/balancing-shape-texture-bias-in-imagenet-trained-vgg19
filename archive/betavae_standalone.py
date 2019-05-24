import os
import subprocess
from utils import *
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from dataset import BaseDataset

from torchvision.utils import make_grid, save_image

from torch.utils.data import DataLoader


ZDIM = 32
NUM_CHANNELS = 3
LEARNING_RATE = 0.0001
BETA = 10
CUDA = True
DEVICE = 'cuda:0' if CUDA else 'cpu'


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor

def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)

# load dataset

class CelebADataset(BaseDataset):

    def __init__(self, root_directory, split='train', transforms=None):
        self.root_directory = root_directory
        super().__init__(root_directory, split, transforms)
        self.INDEX_IMAGE = 1
        self.INDEX_TARGET_IMAGE = 1

    def loadImage(self, filepath):
        return Image.open(filepath)

    def loadDatapoint(self, idx):
        input_filepath = self.datapoints[idx]
        input_image = self.loadImage(input_filepath)

        if self.transforms:
            input_image = self.transforms(input_image)

        return (input_filepath, input_image)

    def loadDataset(self):
        return [ pathJoin(self.root_directory, filename) for filename in os.listdir(self.root_directory) ]


# define model

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
    eps = Variable(std.data.new(std.size()).normal_())
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
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
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
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
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


# train model

def reconstruction_loss(x, x_recon, distribution='gaussian'):
    batch_size = x.size(0)
    assert batch_size != 0
    sigmoid = nn.Sigmoid()

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
    elif distribution == 'gaussian':
        x_recon = sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum').div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def viz_traverse(epoch, model, loader, limit=3, inter=2/3, loc=-1):
    model.eval()
    import random

    decoder = model.decoder
    encoder = model.encoder
    interpolation = torch.arange(-limit, limit+0.1, inter)

    n_dsets = len(loader.dataset)
    rand_idx = random.randint(1, n_dsets-1)

    random_img = loader.dataset.__getitem__(rand_idx)
    with torch.no_grad():
        random_img = Variable(cuda(random_img[1], CUDA)).unsqueeze(0)
    random_img_z = encoder(random_img)[:, :ZDIM]

    with torch.no_grad():
        random_z = Variable(cuda(torch.rand(1, ZDIM), CUDA))

    fixed_idx = 0
    fixed_img = loader.dataset.__getitem__(fixed_idx)
    with torch.no_grad():
        fixed_img = Variable(cuda(fixed_img[1], CUDA)).unsqueeze(0)
    fixed_img_z = encoder(fixed_img)[:, :ZDIM]

    Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

    gifs = []
    for key in Z.keys():
        z_ori = Z[key]
        samples = []
        for row in range(ZDIM):
            if loc != -1 and row != loc:
                continue
            z = z_ori.clone()
            for val in interpolation:
                z[:, row] = val
                sample = F.sigmoid(decoder(z)).data
                samples.append(sample)
                gifs.append(sample)
        samples = torch.cat(samples, dim=0).cpu()
        title = '{}_latent_traversal(iter:{})'.format(key, epoch)

    output_dir = os.path.join('/home/grahman/scratch/unn/betavaeoutputs', str(epoch))
    os.makedirs(output_dir, exist_ok=True)
    gifs = torch.cat(gifs)
    gifs = gifs.view(len(Z), ZDIM, len(interpolation), NUM_CHANNELS, 64, 64).transpose(1, 2)
    for i, key in enumerate(Z.keys()):
        for j, val in enumerate(interpolation):
            save_image(tensor=gifs[i][j].cpu(),
                        filename=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                        nrow=ZDIM, pad_value=1)

        grid2gif(os.path.join(output_dir, key+'*.jpg'),
                    os.path.join(output_dir, key+'.gif'), delay=10)


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),])

    dset = CelebADataset('./space/datasets/CelebA/img_align_celeba', transforms=transform)
    loader = DataLoader(dset,
                       batch_size=32,
                       shuffle=True,
                       num_workers=1,
                       pin_memory=False,
                       drop_last=True)
    
    model = BetaVAE_H(ZDIM, NUM_CHANNELS)
    if CUDA:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for batch_index, batch in enumerate(loader, 1):
        model.train()
        print('Batch {}/{}'.format(batch_index, len(loader)))

        batch_input = batch[loader.dataset.INDEX_IMAGE].to(DEVICE)
        recon, mu, logvar = model(batch_input)
        recon_loss = reconstruction_loss(batch_input, recon)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        beta_vae_loss = recon_loss + (BETA * total_kld)

        optimizer.zero_grad()
        beta_vae_loss.backward()
        optimizer.step()
        print('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                        batch_index, recon_loss.item(), total_kld.item(), mean_kld.item()))

        if ((batch_index + 1) % 1000 == 0):
            viz_traverse(batch_index, model, loader)
