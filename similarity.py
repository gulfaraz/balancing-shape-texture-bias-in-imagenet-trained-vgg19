
# coding: utf-8

# native
import os
import sys

import numpy as np
import pandas as pd

# modules
from utils import *
from dataset import *
from vgg19min import *

# pytorch
import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

np.set_printoptions(threshold=sys.maxsize)

requirements = {
    torch: '1'
}

check_requirements(requirements)


config = configuration()
for k, v in sorted(vars(config).items()):
    print('{0}: {1}'.format(k, v))


IMAGE_SIZE = (config.inputSize, config.inputSize)

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

def load_data(dataset_name, split):
    dataset_path = os.path.join(config.rootPath, 'datasets', dataset_name)

    istrain = split == 'train'
    transforms = train_transforms if istrain else test_transforms

    dataset = MiniImageNetDataset(dataset_path, split=split, transforms=transforms)#raw_transforms)
    loader = DataLoader(dataset, batch_size=config.batchSize, shuffle=istrain, num_workers=config.numberOfWorkers)

    print('{} dataset {} has {} datapoints in {} batches'.format(split, dataset_name, len(dataset), len(loader)))

    return dataset, loader

original_train_dataset, original_train_loader = load_data('miniimagenet', 'train')
original_val_dataset, original_val_loader = load_data('miniimagenet', 'val')

stylized_train_dataset, stylized_train_loader = load_data('stylized-miniimagenet-1.0', 'train')
stylized_val_dataset, stylized_val_loader = load_data('stylized-miniimagenet-1.0', 'val')

for dataset, loader in [
    (original_train_dataset, original_train_loader),
    (original_val_dataset, original_val_loader),
    (stylized_train_dataset, stylized_train_loader),
    (stylized_val_dataset, stylized_val_loader)
]:
    print('{} Datapoints in {} Batches'.format(len(dataset), len(loader)))

dataset_names = [
    'stylized-miniimagenet-1.0', 'stylized-miniimagenet-0.9', 'stylized-miniimagenet-0.8',
    'stylized-miniimagenet-0.7', 'stylized-miniimagenet-0.6', 'stylized-miniimagenet-0.5',
    'stylized-miniimagenet-0.4', 'stylized-miniimagenet-0.3', 'stylized-miniimagenet-0.2',
    'stylized-miniimagenet-0.1', 'stylized-miniimagenet-0.0', 'miniimagenet'
]

def epoch(model, loader, device):
    for batch in loader:
        index_image = loader.dataset.INDEX_IMAGE
        model(batch[index_image].to(device))

checkpoint_map = {
    'miniimagenet_with_in': 'vgg19_in_single_tune_after',
    'stylized_miniimagenet_with_in': 'stylized_vgg19_in_single_tune_after'
}

def load_model(model, model_name):
    checkpoint_path = os.path.join('space', 'models', '{}.ckpt'.format(checkpoint_map[model_name]))
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['weights'])
    model.eval()

def load_run_model_epoch(model_name, loader, dataset_name):
    file_path = os.path.join('csv', '{}-{}'.format(model_name, dataset_name))
    model = create_vgg19_in_bs_single_similarity(filename=file_path)
    if model_name != 'imagenet':
        load_model(model, model_name)
    epoch(model, loader, config.device)

names = ['original_val_loader', 'stylized_val_loader', 'original_train_loader', 'stylized_train_loader']

for index, loader in enumerate([original_val_loader, stylized_val_loader, original_train_loader, stylized_train_loader]):
    load_run_model_epoch('imagenet', loader, names[index])
    load_run_model_epoch('miniimagenet_with_in', loader, names[index])
    load_run_model_epoch('stylized_miniimagenet_with_in', loader, names[index])
