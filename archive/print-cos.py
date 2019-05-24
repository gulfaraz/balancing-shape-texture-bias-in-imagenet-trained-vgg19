
# coding: utf-8

# In[1]:


# native
import sys
import os
from os import listdir
from collections import defaultdict
from PIL import Image
import pprint as pp
import functools
import pickle
import re

# math
import numpy as np
from sklearn.metrics import accuracy_score

# plotting
import matplotlib
from matplotlib import pyplot as plt

# extra
from tqdm import tqdm
import logging

# pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils import model_zoo

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


# In[2]:


requirements = {
    torch: '1',
    matplotlib: '3'
}

def check_requirements(requirements):
    for requirement in requirements:
        error_message = '{} environment does not match requirement'.format(requirement.__name__)
        assert (requirement.__version__[0] == requirements[requirement]), error_message

check_requirements(requirements)


# In[3]:


cuda = torch.cuda.is_available()

if cuda:
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = 'cuda' if cuda else 'cpu'

device


# In[4]:


class PlotGrid:
    def __init__(self, figsize=None):
        self.fig = plt.figure(figsize=figsize)
        self.ax = {}
        self.xlim = {}
        self.ylim = {}
        self.filled = {}
        self.grid = {}
    
    def plot(self, position_id, data, title=None, xlim=None, ylim=None, filled=None, grid=None):
        if position_id in self.ax:
            ax = self.ax[position_id]
        else:
            ax = self.fig.add_subplot(*position_id)

        # cache current values
        if title is None:
            title = ax.get_title()

        if xlim is not None:
            self.xlim[position_id] = xlim

        if ylim is not None:
            self.ylim[position_id] = ylim

        if filled is not None:
            self.filled[position_id] = filled
        
        if position_id not in self.filled:
            self.filled[position_id] = True

        if grid is not None:
            self.grid[position_id] = grid
        
        if position_id not in self.grid:
            self.grid[position_id] = True

        ax.cla()
        ax.clear()
        if type(data).__name__ == 'Image':
            ax.imshow(data)
        else:
            if hasattr(data, 'is_cuda') and data.is_cuda:
                data = data.cpu()
            if hasattr(data, 'numpy'):
                data = data.numpy()
            ax.plot(data)

            if self.filled[position_id]:
                ax.fill_between(range(len(data)), data)

            if self.grid[position_id]:
                ax.grid(True)

            # set xlim
            if position_id in self.xlim:
                ax.set_xlim(*self.xlim[position_id])

            # set ylim
            if position_id in self.ylim:
                ax.set_ylim(*self.ylim[position_id])
        
        # set title
        if title is not None:
            ax.set_title(title)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.ax[position_id] = ax
    
    def prediction_plot(self, position_id, data, title=None, grid=None):
        if position_id in self.ax:
            ax = self.ax[position_id]
        else:
            ax = self.fig.add_subplot(*position_id)

        # cache current values
        if title is None:
            title = ax.get_title()

        if grid is not None:
            self.grid[position_id] = grid
        
        if position_id not in self.grid:
            self.grid[position_id] = True

        ax.cla()
        ax.clear()
        plot_data = data[2]
        plot_labels = data[1]
        if hasattr(plot_data, 'is_cuda') and plot_data.is_cuda:
            plot_data = plot_data.cpu()
        if hasattr(plot_data, 'numpy'):
            plot_data = plot_data.numpy()

        ticks = range(len(plot_data)-1, -1, -1)

        ax.barh(ticks, plot_data, align='center')

        if self.grid[position_id]:
            ax.grid(True)

        # set xlim
        ax.set_xlim(0, 1)

        # set y labels
        ax.set_yticks(ticks)
        ax.set_yticklabels(plot_labels)
        
        # set title
        if title is not None:
            ax.set_title(title)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.ax[position_id] = ax
    
    def savefig(self, filename):
        figure_directory = os.path.join('results', 'activation-plots')
        os.makedirs(figure_directory, exist_ok=True)
        figure_path = os.path.join(figure_directory, filename)
        self.fig.savefig(figure_path, bbox_inches='tight')


# In[5]:


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


toPILImage = transforms.ToPILImage()

softmax = torch.nn.Softmax(dim=1)


# In[6]:


ROOT_PATH = pathJoin(os.sep, 'var', 'node433', 'local', 'gulfaraz')


# In[7]:


class BaseDataset(Dataset):

    def __init__(self, directory, split='train', transforms=None):
        self.datapoints = defaultdict(list)
        self.split = split
        self.directory = pathJoin(directory, split)
        self.datapoints = self.loadDataset()
        self.transforms = transforms

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        datapoint = self.loadDatapoint(idx)
        return datapoint

    def loadDatapoint(self, idx):
        raise NotImplementedError('Function "loadDatapoint" is not implemented')

    def loadDataset(self, name):
        raise NotImplementedError('Function "loadDataset" is not implemented')


# In[8]:


class MiniImageNetDataset(BaseDataset):

    def __init__(self, directory, split='train', transforms=None):
        super().__init__(directory, split, transforms)
        self.descriptions = self.loadDescriptions()
        self.classes = self.loadClasses()
        self.groundtruths = self.loadValidationGroundtruths() if split == 'val' else []
        self.INDEX_IMAGE = 1
        self.INDEX_TARGET = 2
        self.INDEX_LABEL = 3

    def loadDatapoint(self, idx):
        filepath = self.datapoints[idx]
        if not os.path.isfile(filepath):
            filepath = filepath.replace('.JPEG', '.png')
        image = Image.open(filepath).convert('RGB')
        if self.split == 'val':
            groundtruth = self.groundtruths[idx]
        elif self.split == 'train':
            groundtruth = self.classes.index(filepath.split('/').pop().split('_')[0])
        if self.transforms:
            image = self.transforms(image)
        return (filepath, image, groundtruth, self.descriptions[groundtruth])

    def loadDataset(self):
        datapoints = []

        dataset_file_list_filename = '{}.txt'.format(self.split)
        dataset_file_list_path = os.path.join(self.directory, dataset_file_list_filename)

        with open(dataset_file_list_path, 'r') as dataset_file_list_file:
            for line in tqdm(dataset_file_list_file, total=sum(1 for line in open(dataset_file_list_path))):
                file_path = pathJoin(self.directory, self.sanitizeFilename(line))
                datapoints.append(file_path)
        
        return datapoints
    
    def sanitizeFilename(self, filename):
        return filename.replace('"', '').strip()

    def loadDescriptions(self):
        descriptions = []

        descriptions_filename = 'wnids_with_descriptions.txt'
        descriptions_path = pathJoin(self.directory, '..', descriptions_filename)

        with open(descriptions_path, 'r') as descriptions_file:
            for line in descriptions_file:
                description_breakdown = line.split(' ')
                description_breakdown.pop(0)
                description = ' '.join(description_breakdown).strip()
                descriptions.append(description)

        return descriptions

    def loadValidationGroundtruths(self):
        groundtruths = []

        groundtruths_filename = 'val_groundtruth.txt'
        groundtruths_path = pathJoin(self.directory, '..', groundtruths_filename)

        with open(groundtruths_path, 'r') as groundtruths_file:
            for line in groundtruths_file:
                groundtruth_breakdown = line.split(' ')
                groundtruth_breakdown.pop(0)
                groundtruth = ' '.join(groundtruth_breakdown).strip()
                groundtruths.append(int(groundtruth))

        return groundtruths

    def loadClasses(self):
        classes = []

        classes_filename = 'wnids.txt'
        classes_path = pathJoin(self.directory, '..', classes_filename)

        with open(classes_path, 'r') as classes_file:
            for line in classes_file:
                classes.append(line.strip())

        return classes

    def idx2label(self, class_idx):
        return self.classes[class_idx]


# In[9]:


class DeNormalize(object):
    # Source: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = image.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


# In[10]:


TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

imagenet_normalization_values = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

normalize = transforms.Normalize(**imagenet_normalization_values)
denormalize = DeNormalize(**imagenet_normalization_values)


def toImage(tensor_image):
    return toPILImage(denormalize(tensor_image))

raw_transforms = transforms.Compose([
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor()
])

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE[0]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    normalize
])

miniimagenet_dataset_path = os.path.join(ROOT_PATH, 'datasets', 'miniimagenet')

original_train_dataset = MiniImageNetDataset(miniimagenet_dataset_path, transforms=train_transforms)#raw_transforms)
original_val_dataset = MiniImageNetDataset(miniimagenet_dataset_path, split='val', transforms=test_transforms)

original_train_loader = DataLoader(original_train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8)
original_val_loader = DataLoader(original_val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=8)

stylized_miniimagenet_dataset_path = os.path.join(ROOT_PATH, 'datasets', 'stylized-miniimagenet-1.0')

stylized_train_dataset = MiniImageNetDataset(stylized_miniimagenet_dataset_path, transforms=train_transforms)#raw_transforms)
stylized_val_dataset = MiniImageNetDataset(stylized_miniimagenet_dataset_path, split='val', transforms=test_transforms)

stylized_train_loader = DataLoader(stylized_train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8)
stylized_val_loader = DataLoader(stylized_val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=8)

for dataset, loader in [
    (original_train_dataset, original_train_loader),
    (original_val_dataset, original_val_loader),
    (stylized_train_dataset, stylized_train_loader),
    (stylized_val_dataset, stylized_val_loader)
]:
    print('{} Datapoints in {} Batches'.format(len(dataset), len(loader)))


def create_miniimagenet_classifier():
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=4096, out_features=200, bias=True)
    )


# In[14]:


class VGG_COSINE(torch.nn.Module):
    def __init__(self, pretrained=False, eps=torch.tensor(1e-08)):
        super(VGG_COSINE, self).__init__()
        vgg19 = models.vgg19(pretrained=pretrained)
        self.features = vgg19.features
        self.classifier = create_miniimagenet_classifier()
        self.layer_indexes = [1, 6, 11, 20, 29]
        self.eps = eps
#         self.cos = torch.nn.CosineSimilarity(dim=1, eps=eps)

    def forward(self, x):
        similarity_scores = []

        for layer_index, layer in enumerate(self.features):
            x = layer(x)
            if (layer_index in self.layer_indexes):
                similarity_scores.append(self.calculate_similarity_score(x))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        layer_similarity = torch.stack(similarity_scores, dim=1)
#         print('layer_similarity.shape')
#         print(layer_similarity.shape)
#         print(layer_similarity)
        return x, layer_similarity

    def calculate_similarity_score(self, x):
#         torch.set_printoptions(profile="full")
        flat_x = x.view(x.size(0), x.size(1), -1)
        similarity_matrix = self.calculate_cosine_similarity_matrix(flat_x)
        similarity_matrix = similarity_matrix ** 2
#         print('similarity_matrix.shape')
#         print(similarity_matrix.shape)
#         print(similarity_matrix)
        gulfaraz = similarity_matrix.sum(dim=1).sum(dim=1)
        similarity_score = gulfaraz - similarity_matrix.size(1)
        print('similarity_matrix.shape')
        print(similarity_matrix.shape)
        print('similarity_matrix.min()')
        print(similarity_matrix.min())
        print('similarity_matrix.max()')
        print(similarity_matrix.max())
#         print('similarity_score.shape')
#         print(similarity_score.shape)
#         print(similarity_score)
#         torch.set_printoptions(profile="default")
#         raise NotImplemented
        return similarity_score

    def calculate_cosine_similarity_matrix(self, x):
        print('x.shape')
        print(x.shape)
        # https://pytorch.org/docs/stable/nn.html#cosine_similarity
        x_t = x.transpose(1, 2)
        print('x_t.shape')
        print(x_t.shape)
        x_norm = torch.norm(x, dim=2, keepdim=True)
        print('x_norm.shape')
        print(x_norm.shape)
        x_t_norm = x_norm.transpose(1, 2)
        print('x_t_norm.shape')
        print(x_t_norm.shape)
        num = torch.matmul(x, x_t)
        print('num.shape')
        print(num.shape)
        norm_prod = torch.matmul(x_norm, x_t_norm)
        print('norm_prod.shape')
        print(norm_prod.shape)
        den = torch.max(norm_prod, self.eps)
        print('den.shape')
        print(den.shape)
        return num / den


def create_vgg19_cosine_tune_all():
    # load model from pytorch
    vgg19 = VGG_COSINE(pretrained=True)

    # train all layers
    for param in vgg19.parameters():
        param.requires_grad = True

    return vgg19


# In[16]:


def score(prediction, target):
    total = prediction.size(0)
    prediction = prediction.t()
    correct = prediction.eq(target.view(1, -1).expand_as(prediction))
    top1 = correct[:1].view(-1).float().sum(0).item()
    top5 = correct[:5].view(-1).float().sum(0).item()
    return top1, top5, total


def score_value(score, total):
    if total > 0:
        return score/total
    else:
        return 0


def score_model(model, dataloader):
    model.eval()
    total_top1 = 0
    total_top5 = 0
    total_ = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            target = batch[dataloader.dataset.INDEX_TARGET].to(device)
            input = batch[dataloader.dataset.INDEX_IMAGE].to(device)
            output, _ = model(input)
            _, predicted_classes = output.topk(5, 1, True, True)
            top1, top5, total = score(predicted_classes, target)
            total_top1 += top1
            total_top5 += top5
            total_ += total
    return total_top1/total_, total_top5/total_


def calculate_similarity_loss(similarity):
    number_of_kernels = torch.tensor([64., 128., 256., 512., 512.])
    style_weights = torch.tensor([1e3/n**2 for n in number_of_kernels])
    weighted_similarity = style_weights * similarity
    return weighted_similarity.mean() + (number_of_kernels * style_weights).sum()


def create_logger(log_directory, filename, stream=False):
    info_filehandler = logging.FileHandler(os.path.join(log_directory, '{}_info.log'.format(filename)))
    debug_filehandler = logging.FileHandler(os.path.join(log_directory, '{}_debug.log'.format(filename)))

    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    info_filehandler.setFormatter(formatter)
    debug_filehandler.setFormatter(formatter)

    info_filehandler.setLevel(logging.INFO)
    debug_filehandler.setLevel(logging.DEBUG)

    logger = logging.getLogger()
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    if stream:
        streamhandler = logging.StreamHandler(sys.stdout)
        streamhandler.setFormatter(formatter)
        streamhandler.setLevel(logging.DEBUG)
        logger.addHandler(streamhandler)

    logger.addHandler(info_filehandler)
    logger.addHandler(debug_filehandler)

    logger.setLevel(logging.DEBUG)

    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.ERROR)

    return logger


# In[ ]:


supported_models = {
    'vgg19_custom_cosine_similarity_weight_0.08_tune_all_grad_clip_50': create_vgg19_cosine_tune_all
}

for model_type in supported_models:
    print(model_type)
    model = supported_models[model_type]()

    checkpoint_path = pathJoin(ROOT_PATH, 'models', '{}.ckpt'.format(model_type))
    print(checkpoint_path)
    
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        epoch = checkpoint['epoch']
        train_top1_accuracy = checkpoint['train_top1_accuracy']
        train_top5_accuracy = checkpoint['train_top5_accuracy']
        train_loss = checkpoint['train_loss']
        validation_top1_accuracy = checkpoint['validation_top1_accuracy']
        validation_top5_accuracy = checkpoint['validation_top5_accuracy']
        validation_loss = checkpoint['validation_loss']
        model.load_state_dict(checkpoint['weights'])

        model.eval()

        original_top1, original_top5 = score_model(model, original_val_loader)
        # stylized_top1, stylized_top5 = score_model(model, stylized_val_loader)
        print('Original: Epoch: {} Top1: {:.4f} Top5: {:.4f}'.format(epoch, original_top1, original_top5))
        # print('Stylized: Epoch: {} Top1: {:.4f} Top5: {:.4f}'.format(epoch, stylized_top1, stylized_top5))
        print('Train: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(train_loss, train_top1_accuracy, train_top5_accuracy))
        print('Validation: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(validation_loss, validation_top1_accuracy, validation_top5_accuracy))
    else:
        print('Checkpoint not available for model {}'.format(model_type))
    del model
    torch.cuda.empty_cache()
