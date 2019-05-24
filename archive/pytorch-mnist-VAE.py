
# coding: utf-8

# In[1]:

# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torchvision.utils import save_image

test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

DEBUG = False

bs = 256
zdim = 20

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/mnist/', train=True, transform=test_transforms, download=True)
test_dataset = datasets.MNIST(root='./data/mnist/', train=False, transform=test_transforms, download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, num_workers=8)


# In[2]:

class VAE_VGG19(nn.Module):
    def __init__(self, n_channels, z_dim):
        super(VAE_VGG19, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.encoder = torch.nn.ModuleList([])
        self.decoder = torch.nn.ModuleList([])
        for layer in vgg19.features:
            self.encoder.append(self.get_encoding_layer(layer))
            self.decoder.insert(0, self.get_decoding_layer(layer))
        self.decoder.append(nn.Sigmoid())
        # encoder part
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(n_channels, 32, kernel_size=3),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(32, 32, kernel_size=3),
        #     nn.LeakyReLU(0.2)
        # )
        
        feature_size = 25088
        
        self.encoder_to_latent = nn.Linear(feature_size, z_dim)

        self.latent_to_mu = nn.Linear(z_dim, z_dim)
        self.latent_to_logvar = nn.Linear(z_dim, z_dim)

        # decoder part
        self.latent_to_decoder = nn.Linear(z_dim, feature_size)
        
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(32, 32, kernel_size=3),
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(32, n_channels, kernel_size=3),
        #     nn.LeakyReLU(0.2),
        #     nn.Sigmoid()
        # )
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
    
    def forward(self, x):
        # encode
#         print('x.shape {}'.format(x.shape))
        encoded_x, indices = self.encode(x)
        feature_shape = encoded_x.shape
#         print('encoded_x.shape {}'.format(encoded_x.shape))
        encoded_x = encoded_x.view(encoded_x.size(0), -1)
#         print('flat encoded_x.shape {}'.format(encoded_x.shape))
        z = self.encoder_to_latent(encoded_x)
#         print('z.shape {}'.format(z.shape))
        
        # sample
        mu = self.latent_to_mu(z)
#         print('mu.shape {}'.format(mu.shape))
        log_var = self.latent_to_logvar(z)
#         print('log_var.shape {}'.format(log_var.shape))
        z = self.sampling(mu, log_var)
#         print('z.shape {}'.format(z.shape))
        
        # decode
        feature = self.latent_to_decoder(z)
        feature = feature.view(*feature_shape)
#         print('feature.shape {}'.format(feature.shape))
        decoded_x = self.decode(feature, indices=indices)
        return decoded_x, mu, log_var

    def get_encoding_layer(self, layer):
        if isinstance(layer, torch.nn.MaxPool2d):
            return torch.nn.MaxPool2d(
                layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                return_indices=True,
                ceil_mode=layer.ceil_mode
            )
        return layer

    def get_decoding_layer(self, layer):
        if isinstance(layer, torch.nn.Conv2d):
            return torch.nn.ConvTranspose2d(
                layer.out_channels,
                layer.in_channels,
                layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=(layer.bias is not None)
            )
        elif isinstance(layer, torch.nn.MaxPool2d):
            return torch.nn.MaxUnpool2d(
                layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding
            )
        return layer
    
    def encode(self, x):
        indices = []
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, torch.nn.MaxPool2d):
                indices.append(x[1])
                x = x[0]
        return x, indices
    
    def decode(self, x, indices=None):
        for layer in self.decoder:
            if isinstance(layer, torch.nn.MaxUnpool2d):
                if indices is None:
                    upsample = torch.nn.Upsample(scale_factor=2)
                    maxpool = torch.nn.MaxPool2d(
                        layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                        return_indices=True
                    )
                    _, index = maxpool(torch.zeros_like(upsample(x)))
                else:
                    index = indices.pop()
                x = layer(x, index)
            else:
                x = layer(x)
        return x


class VAE_CNN(nn.Module):
    def __init__(self, n_channels, z_dim):
        super(VAE_CNN, self).__init__()
        
        # encoder part
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.LeakyReLU(0.2)
        )
        
        feature_size = 18432
        
        self.encoder_to_latent = nn.Linear(feature_size, z_dim)

        self.latent_to_mu = nn.Linear(z_dim, z_dim)
        self.latent_to_logvar = nn.Linear(z_dim, z_dim)

        # decoder part
        self.latent_to_decoder = nn.Linear(z_dim, feature_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, n_channels, kernel_size=3),
            nn.LeakyReLU(0.2),
            nn.Sigmoid()
        )
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
    
    def forward(self, x):
        # encode
#         print('x.shape {}'.format(x.shape))
        encoded_x = self.encoder(x)
        feature_shape = encoded_x.shape
#         print('encoded_x.shape {}'.format(encoded_x.shape))
        encoded_x = encoded_x.view(encoded_x.size(0), -1)
#         print('flat encoded_x.shape {}'.format(encoded_x.shape))
        z = self.encoder_to_latent(encoded_x)
#         print('z.shape {}'.format(z.shape))
        
        # sample
        mu = self.latent_to_mu(z)
#         print('mu.shape {}'.format(mu.shape))
        log_var = self.latent_to_logvar(z)
#         print('log_var.shape {}'.format(log_var.shape))
        z = self.sampling(mu, log_var)
#         print('z.shape {}'.format(z.shape))
        
        # decode
        feature = self.latent_to_decoder(z)
        feature = feature.view(*feature_shape)
#         print('feature.shape {}'.format(feature.shape))
        decoded_x = self.decoder(feature)
        return decoded_x, mu, log_var


class VAE_VECTOR(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE_VECTOR, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

# build model
# vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
vae = VAE_VGG19(n_channels=1, z_dim=zdim)
device = 'cpu'
if torch.cuda.is_available():
    vae.cuda()
    device = 'cuda:0'


# In[3]:

print(vae)


# In[4]:

optimizer = optim.Adam(vae.parameters())

bce_loss = nn.BCELoss()
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    # print('recon_x {}'.format(recon_x.shape))
    # print('x {}'.format(x.shape))
    # print('recon_x {}'.format(recon_x.is_cuda))
    # print('x {}'.format(x.is_cuda))
    print('recon_x {}'.format(recon_x[0, 0, :, :]))
    print('x {}'.format(x[0, 0, :, :]))
    BCE = bce_loss(recon_x, x)
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE, KLD


# In[5]:

def train(epoch, beta):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        bce, kld = loss_function(recon_batch, data, mu, log_var)
        beta_kld = kld * beta
        loss = bce + beta_kld
        print('loss {} bce {} beta_kld {} kld {} beta {}'.format(loss, bce, beta_kld, kld, beta))
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
            if DEBUG:
                break
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


# In[6]:

def test(beta):
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            bce, kld = loss_function(recon, data, mu, log_var)
            beta_kld = beta * kld
            test_loss += (bce + beta_kld).item()
            print('loss {} bce {} beta_kld {} kld {} beta{}'.format(test_loss, bce, beta_kld, kld, beta))
            if DEBUG:
                break
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

# In[7]:

def generate(vae, epoch):
    with torch.no_grad():
        z = torch.randn(64, zdim).to(device)
        feature = vae.latent_to_decoder(z)
        feature = feature.view(64, 512, 7, 7)
        # feature = feature.view(64, 32, 24, 24)
        sample = vae.decode(feature).to(device)
        # sample = vae.decoder(feature).to(device)

        save_image(sample.view(64, 3, 224, 224), './samples/sample_' + str(epoch) + '.png')
        # save_image(sample.view(64, 1, 28, 28), './samples/sample_' + '.png')


# In[8]:

beta_min = 0.0
beta_max = 1.0

# vector
# 1 does NOT work
# 0.1 does NOT work
# 0.01 works
# 0.001 works
# 0.0001 works

# conv
# 1 does NOT work
# 0.1 does NOT work
# 0.05 barely works
# 0.01 almost works
# 0.001 does NOT work
# 0.0001 does NOT work

beta_mul = 0.05
beta_step = beta_min
beta = beta_min

n_epochs = 10

print('Starting Training...')
for epoch in range(1, n_epochs + 1):
    print('Epoch {} Started'.format(epoch))
    anneal_rate = (1.0 - beta_step) / n_epochs
    train(epoch, beta)
    test(beta)
    beta_step += anneal_rate
    beta = beta_step * beta_mul
    generate(vae, epoch)
    print('Epoch {} Complete'.format(epoch))
