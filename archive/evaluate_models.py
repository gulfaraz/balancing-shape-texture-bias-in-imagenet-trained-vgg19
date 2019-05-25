
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
'''
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
'''

# In[11]:


# In[12]:

class InstanceNormBatchSwap(torch.nn.Module):
    def __init__(self, n_neurons, affine=False, eps=1e-5):
        super(InstanceNormBatchSwap, self).__init__()
        self.n_neurons = n_neurons
        self.eps = eps
        assert affine == False, 'affine parameters not implemented'

    def forward(self, input):
        assert input.shape[1] == self.n_neurons, "Input has incorrect shape"

        temp = input.view(input.size(0), input.size(1), -1)
        mean = temp.mean(2, keepdim=True).unsqueeze(-1)
        std = temp.std(2, keepdim=True).unsqueeze(-1)
        den = torch.sqrt(std.pow(2) + self.eps)
        output = (input - mean)/den
        indices = torch.randperm(input.size(0))
        output = output * std.index_select(0, indices) + mean.index_select(0, indices)

        return output
    
    def __repr__(self):
        return 'InstanceNormBatchSwap({}, eps={})'.format(self.n_neurons, self.eps)


# In[13]:


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


class VGG_IN(torch.nn.Module):
    def __init__(self, layer_index, instance_normalization_function=torch.nn.InstanceNorm2d, affine=False, pretrained=False):
        super(VGG_IN, self).__init__()
        vgg19 = models.vgg19(pretrained=pretrained)
        self.features1 = vgg19.features[:layer_index]
        self.instance_normalization = instance_normalization_function(vgg19.features[layer_index].out_channels, affine=affine)
        self.features2 = vgg19.features[layer_index:]
        self.classifier = create_miniimagenet_classifier()

    def forward(self, x):
        x = self.features1(x)
        x = self.instance_normalization(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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
        similarity_score = similarity_matrix.sum(dim=1).sum(dim=1) - similarity_matrix.size(1)
#         print('similarity_score.shape')
#         print(similarity_score.shape)
#         print(similarity_score)
#         torch.set_printoptions(profile="default")
#         raise NotImplemented
        return similarity_score

    def calculate_cosine_similarity_matrix(self, x):
#         print('x.shape')
#         print(x.shape)
        # https://pytorch.org/docs/stable/nn.html#cosine_similarity
        x_t = x.transpose(1, 2)
#         print('x_t.shape')
#         print(x_t.shape)
        x_norm = torch.norm(x, dim=2, keepdim=True)
#         print('x_norm.shape')
#         print(x_norm.shape)
        x_t_norm = x_norm.transpose(1, 2)
#         print('x_t_norm.shape')
#         print(x_t_norm.shape)
        num = torch.matmul(x, x_t)
#         print('num.shape')
#         print(num.shape)
        norm_prod = torch.matmul(x_norm, x_t_norm)
#         print('norm_prod.shape')
#         print(norm_prod.shape)
        den = torch.max(norm_prod, self.eps)
#         print('den.shape')
#         print(den.shape)
        return num / den


class VGG_COSINE_ZEROS(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(VGG_COSINE_ZEROS, self).__init__()
        vgg19 = models.vgg19(pretrained=pretrained)
        self.features = vgg19.features
        self.classifier = create_miniimagenet_classifier()
        self.layer_indexes = [1, 6, 11, 20, 29]

    def forward(self, x):
        similarity_scores = []

        for layer_index, layer in enumerate(self.features):
            x = layer(x)
            if (layer_index in self.layer_indexes):
                similarity_score = self.similarity_matrix(x)
                batch_similarity = similarity_score.mean(dim=1).mean(dim=1)
                similarity_scores.append(torch.zeros_like(batch_similarity))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x, torch.stack(similarity_scores, dim=1)

    def similarity_matrix(self, x):
        flat_x = x.view(x.size(0), x.size(1), -1)
        similarity_score = self.cosine_similarity(flat_x)
        return similarity_score

    def cosine_similarity(self, x, eps=torch.tensor(1e-8)):
        x_norm = x / torch.norm(x, dim=2, keepdim=True)
        return 1 - torch.max(torch.matmul(x_norm, x_norm.transpose(1, 2)), eps)


# In[15]:


def create_vgg19_in_pretrained():
    vgg = VGG_IN(21, pretrained=True)

    # freeze layers before IN
    for param in vgg.features1.parameters():
        param.requires_grad = False

    # train layers after IN
    for param in vgg.features2.parameters():
        param.requires_grad = True

    # train fc layers
    for param in vgg.classifier.parameters():
        param.requires_grad = True

    return vgg


def create_vgg19_scratch():
    # load model from pytorch
    vgg19 = models.vgg19(pretrained=False)

    vgg19.classifier = create_miniimagenet_classifier()

    vgg19.apply(init_weights)

    # train all layers
    for param in vgg19.parameters():
        param.requires_grad = True

    return vgg19


def create_vgg19_pretrained():
    # load model from pytorch
    vgg19 = models.vgg19(pretrained=True)

    # freeze cnn layers
    for param in vgg19.parameters():
        param.requires_grad = False

    vgg19.classifier = create_miniimagenet_classifier()

    # train fc layers
    for param in vgg19.classifier.parameters():
        param.requires_grad = True

    return vgg19


def create_vgg19_in_affine_single_tune_all():
    vgg = VGG_IN(21, affine=True, pretrained=True)

    # train all layers
    for param in vgg.parameters():
        param.requires_grad = True

    return vgg


def create_vgg19_bn_all_tune_all():
    vgg19 = torchvision.models.vgg19_bn(pretrained=True)

    # train cnn layers
    for param in vgg19.features.parameters():
        param.requires_grad = True

    vgg19.classifier = create_miniimagenet_classifier()

    # train fc layers
    for param in vgg19.classifier.parameters():
        param.requires_grad = True

    return vgg19


def create_vgg19_cosine_tune_all():
    # load model from pytorch
    vgg19 = VGG_COSINE(pretrained=True)

    # train all layers
    for param in vgg19.parameters():
        param.requires_grad = True

    return vgg19

'''

FROM HERE

'''

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def create_vgg19_in_bs_eval():
    vgg = VGG_IN_(21, pretrained=True)

    # freeze layers before IN
    for param in vgg.features1.parameters():
        param.requires_grad = False

    # train layers after IN
    for param in vgg.features2.parameters():
        param.requires_grad = True

    # train fc layers
    for param in vgg.classifier.parameters():
        param.requires_grad = True

    return vgg


def create_vgg19_in_pretrained():
    vgg = VGG_IN(21, pretrained=True)

    # freeze layers before IN
    for param in vgg.features1.parameters():
        param.requires_grad = False

    # train layers after IN
    for param in vgg.features2.parameters():
        param.requires_grad = True

    # train fc layers
    for param in vgg.classifier.parameters():
        param.requires_grad = True

    return vgg


def create_vgg19_in_single_tune_all():
    vgg = VGG_IN(21, pretrained=True)

    # train all layers
    for param in vgg.parameters():
        param.requires_grad = True

    return vgg


def create_vgg19_in_affine_single_tune_all():
    vgg = VGG_IN(21, affine=True, pretrained=True)

    # train all layers
    for param in vgg.parameters():
        param.requires_grad = True

    return vgg


def create_vgg19_in_batch_stats_pretrained():
    vgg = VGG_IN(21, instance_normalization_function=InstanceNormBatchSwap, pretrained=True)

    # freeze layers before IN
    for param in vgg.features1.parameters():
        param.requires_grad = False

    # train layers after IN
    for param in vgg.features2.parameters():
        param.requires_grad = True

    # train fc layers
    for param in vgg.classifier.parameters():
        param.requires_grad = True

    return vgg


def create_vgg19_in_bs_single_tune_all():
    vgg = VGG_IN(21, instance_normalization_function=InstanceNormBatchSwap, pretrained=True)

    # train all layers
    for param in vgg.parameters():
        param.requires_grad = True

    return vgg


def create_vgg19_bn_all_tune_fc():
    vgg19 = torchvision.models.vgg19_bn(pretrained=True)

    # freeze cnn layers
    for param in vgg19.features.parameters():
        param.requires_grad = False

    vgg19.classifier = create_miniimagenet_classifier()

    # train fc layers
    for param in vgg19.classifier.parameters():
        param.requires_grad = True

    return vgg19


def create_vgg19_bn_all_tune_all():
    vgg19 = torchvision.models.vgg19_bn(pretrained=True)

    # train cnn layers
    for param in vgg19.features.parameters():
        param.requires_grad = True

    vgg19.classifier = create_miniimagenet_classifier()

    # train fc layers
    for param in vgg19.classifier.parameters():
        param.requires_grad = True

    return vgg19


def create_vgg19_in_all_tune_all_root(instance_normalization_function=torch.nn.InstanceNorm2d):
    vgg19_in = torchvision.models.vgg19_bn(pretrained=False)
    vgg19 = torchvision.models.vgg19(pretrained=True)

    transfer_layer_index = 0
    for layer_index, layer in enumerate(vgg19_in.features):
        # replace batch norm with instance norm
        if isinstance(layer, torch.nn.BatchNorm2d):
            vgg19_in.features[layer_index] = instance_normalization_function(layer.num_features)
    
        # transfer conv weights
        if isinstance(layer, torch.nn.Conv2d):
            vgg19_in.features[layer_index].load_state_dict(vgg19.features[transfer_layer_index].state_dict())

        if (isinstance(layer, torch.nn.ReLU) or
            isinstance(layer, torch.nn.MaxPool2d) or
            isinstance(layer, torch.nn.Conv2d)):
            transfer_layer_index += 1

    vgg19_in.classifier = create_miniimagenet_classifier()

    # train all layers
    for param in vgg19_in.parameters():
        param.requires_grad = True

    return vgg19_in


def create_vgg19_in_all_tune_all():
    return create_vgg19_in_all_tune_all_root()


def create_vgg19_in_bs_all_tune_all():
    return create_vgg19_in_all_tune_all_root(instance_normalization_function=InstanceNormBatchSwap)


def create_vgg19_pretrained():
    # load model from pytorch
    vgg19 = models.vgg19(pretrained=True)

    # freeze cnn layers
    for param in vgg19.parameters():
        param.requires_grad = False

    vgg19.classifier = create_miniimagenet_classifier()

    # train fc layers
    for param in vgg19.classifier.parameters():
        param.requires_grad = True

    return vgg19


def create_vgg19_scratch():
    # load model from pytorch
    vgg19 = models.vgg19(pretrained=False)

    vgg19.classifier = create_miniimagenet_classifier()

    vgg19.apply(init_weights)

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


def score_similarity_model(model, dataloader):
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


def score_model(model, dataloader):
    model.eval()
    total_top1 = 0
    total_top5 = 0
    total_ = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            target = batch[dataloader.dataset.INDEX_TARGET].to(device)
            input = batch[dataloader.dataset.INDEX_IMAGE].to(device)
            output = model(input)
            _, predicted_classes = output.topk(5, 1, True, True)
            top1, top5, total = score(predicted_classes, target)
            total_top1 += top1
            total_top5 += top5
            total_ += total
    return total_top1/total_, total_top5/total_


def calculate_similarity_loss(similarity):
    style_weights = torch.tensor([1e3/n**2 for n in [64, 128, 256, 512, 512]])
    weighted_similarity = style_weights * similarity
    return weighted_similarity.mean()


# def calculate_similarity_loss(similarity):
#     return similarity.mean()


# In[17]:


# In[18]:


# In[ ]:


# In[ ]:


# In[ ]:


supported_models = {
#     'vgg19_in_batch_stats_pretrained_in_tuned_min': create_vgg19_in_pretrained, # create_vgg19_in_batch_stats_pretrained / create_vgg19_in_bs_eval / create_vgg19_in_pretrained
#     'vgg19_scratch_min': create_vgg19_scratch,
#     'vgg19_cosine_tune_all_no_similarity': create_vgg19_cosine_tune_all
    # 'vgg19_in_pretrained_in_tuned_min_old': create_vgg19_in_pretrained, # VGG19_TUNE_FC
    # 'vgg19_scratch_min_backup': create_vgg19_scratch,
    # 'vgg19_in_pretrained_in_tuned_min': create_vgg19_in_pretrained, # VGG19_TUNE_FC
    # 'vgg19_in_batch_stats_pretrained_in_tuned_min': create_vgg19_in_pretrained,
    # 'vgg19_pretrained_in_tuned_min': create_vgg19_pretrained,
    # 'vgg19_bn_pretrained_in_tuned_min': create_vgg19_bn_all_tune_fc,
    # 'vgg19_in_single_tune_all_sgd': create_vgg19_in_single_tune_all,
    # 'vgg19_in_single_tune_all': create_vgg19_in_single_tune_all,
    # 'vgg19_in_bs_single_tune_all': create_vgg19_in_bs_single_tune_all,
    # 'vgg19_in_all_tune_all': create_vgg19_in_all_tune_all,
    # 'vgg19_in_bs_all_tune_all': create_vgg19_in_bs_all_tune_all,
    # 'vgg19_cosine_tune_all': create_vgg19_cosine_tune_all, # VGG19_COSINE_ALL_TUNE_ALL
    # 'vgg19_in_single_tune_all_2': create_vgg19_in_single_tune_all,
    # 'vgg19_in_affine_single_tune_all': create_vgg19_in_affine_single_tune_all, # VGG19_IN_AFFINE_SINGLE_TUNE_ALL
    # 'vgg19_bn_all_tune_all': create_vgg19_bn_all_tune_all,
    # 'vgg19_cosine_tune_all_no_similarity_loss': create_vgg19_cosine_tune_all,
    # 'with_sim_weight_0_with_sim_loss': create_vgg19_cosine_tune_all,
    # 'vgg19_custom_cosine_similarity_weight_0.02_tune_all': create_vgg19_cosine_tune_all,
    # 'with_sim_weight_20_with_sim_loss': create_vgg19_cosine_tune_all,
    # 'vgg19_custom_cosine_similarity_weight_10_tune_all_grad_clip_50_epoch_34': create_vgg19_cosine_tune_all,
    # 'vgg19_custom_cosine_similarity_weight_10_tune_all_grad_clip_50': create_vgg19_cosine_tune_all,
    # 'stylized_vgg19_scratch_min_20': create_vgg19_scratch,
    # 'stylized_vgg19_scratch_min_backup': create_vgg19_scratch,
    # 'stylized_vgg19_in_pretrained_in_tuned_min': create_vgg19_in_pretrained,
    # 'stylized_vgg19_in_batch_stats_pretrained_in_tuned_min': create_vgg19_in_pretrained,
    # 'stylized_vgg19_pretrained_in_tuned_min': create_vgg19_pretrained,
    'vgg19_custom_cosine_similarity_weight_0.04_tune_all_grad_clip_50_pos_loss': create_vgg19_cosine_tune_all,
}

# for model_type in supported_models:
#     print(model_type)
#     model = supported_models[model_type]()
# #     print(model)
#     for batch in original_train_loader:
#         index_image = original_train_loader.dataset.INDEX_IMAGE
#         model(batch[index_image].to(device))
#         break
#     del model
#     torch.cuda.empty_cache()


# In[ ]:


# In[ ]:


# ## Check Performance

# In[ ]:

sim_models = [
    'vgg19_cosine_tune_all',
    'vgg19_cosine_tune_all_no_similarity_loss',
    'with_sim_weight_0_with_sim_loss',
    'vgg19_custom_cosine_similarity_weight_0.02_tune_all',
    'with_sim_weight_20_with_sim_loss',
    'vgg19_custom_cosine_similarity_weight_10_tune_all_grad_clip_50_epoch_34',
    'vgg19_custom_cosine_similarity_weight_10_tune_all_grad_clip_50',
    'vgg19_custom_cosine_similarity_weight_0.04_tune_all_grad_clip_50_pos_loss'
    ]

dataset_names = [
    'miniimagenet',
    'stylized-miniimagenet-0.1',
    'stylized-miniimagenet-0.2',
    'stylized-miniimagenet-0.3',
    'stylized-miniimagenet-0.4',
    'stylized-miniimagenet-0.5',
    'stylized-miniimagenet-0.6',
    'stylized-miniimagenet-0.7',
    'stylized-miniimagenet-0.8',
    'stylized-miniimagenet-0.9',
    'stylized-miniimagenet-1.0'
    ]

def eval_model_dataset(model, dataset_name, scoring_function):
    dataset_path = os.path.join(ROOT_PATH, 'datasets', dataset_name)
    val_dataset = MiniImageNetDataset(dataset_path, split='val', transforms=test_transforms)
    val_loader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=8)
    print('Dataset {} has {} Datapoints in {} Batches'.format(dataset_name, len(val_dataset), len(val_loader)))
    top1, top5 = scoring_function(model, val_loader)
    print('{}: Epoch: {} Top1: {:.4f} Top5: {:.4f}'.format(dataset_name, epoch, top1, top5))

for model_type in supported_models:
    print(model_type)
    model = supported_models[model_type]()
    
#     checkpoint = {
#         'epoch': epoch,
#         'train_accuracy': train_accuracy,
#         'train_loss': train_loss,
#         'validation_accuracy': validation_accuracy,
#         'weights': model.state_dict()
#     }
    checkpoint_path = pathJoin(ROOT_PATH, 'models', '{}.ckpt'.format(model_type))
#     checkpoint_path = pathJoin('trained_models', 'min', 'stylized_{}.ckpt'.format(model_type))
    
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        # for key in checkpoint:
        #     print(key)

        epoch = checkpoint['epoch']
        if 'train_top1_accuracy' in checkpoint:
            train_top1_accuracy = checkpoint['train_top1_accuracy']
            train_top5_accuracy = checkpoint['train_top5_accuracy']
        else:
            train_top1_accuracy = checkpoint['train_accuracy']
            train_top5_accuracy = checkpoint['train_accuracy']
        train_loss = checkpoint['train_loss']

        if 'validation_top1_accuracy' in checkpoint:
            validation_top1_accuracy = checkpoint['validation_top1_accuracy']
            validation_top5_accuracy = checkpoint['validation_top5_accuracy']
        else:
            validation_top1_accuracy = checkpoint['validation_accuracy']
            validation_top5_accuracy = checkpoint['validation_accuracy']
        validation_loss = 0.0
        if 'validation_loss' in checkpoint:
            validation_loss = checkpoint['validation_loss']
        model.load_state_dict(checkpoint['weights'])

        model.eval()

        scoring_function = score_model
        if model_type in sim_models:
            scoring_function = score_similarity_model

        for dataset_name in dataset_names:
            eval_model_dataset(model, dataset_name, scoring_function)
        print('Train: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(train_loss, train_top1_accuracy, train_top5_accuracy))
        print('Validation: Loss: {:.4f} Top1 Accuracy: {:.4f} Top5 Accuracy: {:.4f}'.format(validation_loss, validation_top1_accuracy, validation_top5_accuracy))
    else:
        print('Checkpoint not available for model {}'.format(model_type))
    del model
    torch.cuda.empty_cache()

