import os
import argparse
import torch
import torchvision.transforms as transforms

def check_requirements(requirements):
    for requirement in requirements:
        error_message = '{} environment does not match requirement'.format(requirement.__name__)
        assert (requirement.__version__[0] == requirements[requirement]), error_message


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
    parser.add_argument('--numberOfEpochs', type=int, default=50,
                        help='number of epochs for training')
    parser.add_argument('--batchSize', type=int, default=32,
                        help='batch size for training')

    parser.add_argument('--train', action='store_false', default=True,
                        help='train the models')

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
