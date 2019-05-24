import math
import os
import argparse
import subprocess
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

def check_requirements(requirements):
    for requirement in requirements:
        error_message = '{} environment does not match requirement'.format(requirement.__name__)
        assert (requirement.__version__[0] == requirements[requirement]), error_message

def roundUp(x, d=100):
    return int(math.ceil(x/d)) * d

def pathJoin(*args):
    return os.path.abspath(os.path.join(*args))


def pprint(*args):
    pp.pprint(*args)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def smooth(x, span=10):
    return [ np.mean(x[i:i+span]) for i in range(len(x) - span + 1)]


def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif + ' && rm ' + image_str
    subprocess.call(str1, shell=True)


def explore_betavae(model_name, model, image_directory, epoch, loader, device, limit=3, inter=2/3, loc=-1):
    model.eval()
    import random

    z_dim = model.z_dim
    decoder = model.decoder
    encoder = model.encoder
    interpolation = torch.arange(-limit, limit+0.1, inter)

    n_dsets = len(loader.dataset)
    rand_idx = random.randint(1, n_dsets-1)

    random_img = loader.dataset.__getitem__(rand_idx)
    with torch.no_grad():
        random_img = torch.Tensor(random_img[loader.dataset.INDEX_IMAGE].to(device)).unsqueeze(0)
    random_img_z = encoder(random_img.to(device))[:, :z_dim]

    with torch.no_grad():
        random_z = torch.Tensor(torch.rand(1, z_dim))

    fixed_idx = 0
    fixed_img = loader.dataset.__getitem__(fixed_idx)
    with torch.no_grad():
        fixed_img = torch.Tensor(fixed_img[loader.dataset.INDEX_IMAGE].to(device)).unsqueeze(0)
    fixed_img_z = encoder(fixed_img.to(device))[:, :z_dim]

    Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

    gifs = []
    for key in Z.keys():
        z_ori = Z[key]
        samples = []
        for row in range(z_dim):
            if loc != -1 and row != loc:
                continue
            z = z_ori.clone()
            for val in interpolation:
                z[:, row] = val
                sample = torch.sigmoid(decoder(z)).data
                samples.append(sample)
                gifs.append(sample)
        samples = torch.cat(samples, dim=0).cpu()
        title = '{}_latent_traversal(iter:{})'.format(key, epoch)

    os.makedirs(image_directory, exist_ok=True)
    gifs = torch.cat(gifs)
    gifs = gifs.view(len(Z), z_dim, len(interpolation), 3, 128, 128).transpose(1, 2)
    for i, key in enumerate(Z.keys()):
        for j, val in enumerate(interpolation):
            save_image(tensor=gifs[i][j].cpu(),
                        filename=os.path.join(image_directory, '{}_epoch_{}_{}.jpg'.format(key, epoch, j)),
                        nrow=z_dim//4, pad_value=1)

        grid2gif(os.path.join(image_directory, '{}_epoch_{}_*.jpg'.format(key, epoch)),
                    os.path.join(image_directory, '{}_epoch_{}.gif'.format(key, epoch)), delay=10)


toPILImage = transforms.ToPILImage()


softmax = torch.nn.Softmax(dim=1)


def configuration():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General arguments
    parser.add_argument('--rootPath', type=str, default=pathJoin(os.sep, 'var', 'node433', 'local', 'gulfaraz'),
                        help='output path')
    parser.add_argument('--numberOfWorkers', type=int, default=8,
                        help='number of threads used by data loader')

    parser.add_argument('--disableCuda', action='store_true',
                        help='disable the use of CUDA')
    parser.add_argument('--cudaDevice', type=int, default=0,
                        help='specify which GPU to use')
    parser.add_argument('--torchSeed', type=int,
                        help='set a torch seed', default=42)

    parser.add_argument('--inputSize', type=int, default=224,
                        help='extent of input layer in the network')
    parser.add_argument('--vaeImageSize', type=int, default=128,
                        help='extent of input and target layer in the autoencoder')
    parser.add_argument('--numberOfEpochs', type=int, default=50,
                        help='number of epochs for training')
    parser.add_argument('--batchSize', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--learningRate', type=float, default=0.0001,
                        help='learning rate for training')
    parser.add_argument('--autoencoderLearningRate', type=float, default=0.001,
                        help='learning rate for autoencoder training')
    parser.add_argument('--classifierLearningRate', type=float, default=0.001,
                        help='learning rate for classifier training')
    parser.add_argument('--beta', type=float, default=0.2,
                        help='beta value for the betavae loss')
    parser.add_argument('--zdim', type=int, default=32,
                        help='latent space dimension size for the betavae')

    parser.add_argument('--train', action='store_true', default=False,
                        help='train the models')

    parser.add_argument('--exists', action='store_true', default=False,
                        help='check if the trained models exist')

    parser.add_argument('--model', action='append', type=str,
                        default=None,
                        help='name of model(s)')

    args = parser.parse_args()

    arg_vars = vars(args)

    if args.torchSeed is not None:
        torch.manual_seed(arg_vars['torchSeed'])
    else:
        arg_vars['torchSeed'] = torch.initial_seed()

    if torch.cuda.is_available() and not arg_vars['disableCuda']:
        torch.backends.cudnn.benchmark = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        arg_vars['device'] = torch.device(
            'cuda:{}'.format(arg_vars['cudaDevice']))
    else:
        arg_vars['device'] = torch.device('cpu')

    return args

