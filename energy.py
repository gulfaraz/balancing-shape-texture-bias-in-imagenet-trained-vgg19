
# coding: utf-8

# In[1]:

import os
import cv2
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from utils import *
from dataset import *

# pytorch
import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

# img_path = 'img.jpg' ## Choose image
# img_path = 'elephant.jpg' ## Choose image

config = configuration()

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
    transforms.Resize(roundUp(IMAGE_SIZE[0])),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    normalize
])

def load_data(dataset_name, split):
    dataset_path = os.path.join(config.rootPath, 'datasets', dataset_name)

    istrain = split == 'train'
    transforms = train_transforms if istrain else test_transforms

    dataset = ImageNet200Dataset(dataset_path, split=split, transforms=transforms)#raw_transforms)
    loader = DataLoader(dataset, batch_size=config.batchSize, shuffle=istrain, num_workers=config.numberOfWorkers)

    print('{} dataset {} has {} datapoints in {} batches'.format(split, dataset_name, len(dataset), len(loader)))

    return dataset, loader

# nonstylized_train_dataset, nonstylized_train_loader = load_data('imagenet200', 'train')
nonstylized_val_dataset, nonstylized_val_loader = load_data('imagenet200', 'val')

# stylized_train_dataset, stylized_train_loader = load_data('stylized-imagenet200-1.0', 'train')
stylized_val_dataset, stylized_val_loader = load_data('stylized-imagenet200-1.0', 'val')

for dataset, loader in [
    # (nonstylized_train_dataset, nonstylized_train_loader),
    (nonstylized_val_dataset, nonstylized_val_loader),
    # (stylized_train_dataset, stylized_train_loader),
    (stylized_val_dataset, stylized_val_loader)
]:
    print('{} Datapoints in {} Batches'.format(len(dataset), len(loader)))

dataset_names = [
    'stylized-imagenet200-1.0', 'stylized-imagenet200-0.9', 'stylized-imagenet200-0.8',
    'stylized-imagenet200-0.7', 'stylized-imagenet200-0.6', 'stylized-imagenet200-0.5',
    'stylized-imagenet200-0.4', 'stylized-imagenet200-0.3', 'stylized-imagenet200-0.2',
    'stylized-imagenet200-0.1', 'stylized-imagenet200-0.0', 'imagenet200'
]

filter_size = 2 ## Set Filter Size


# In[2]:


def load_image(image_path):
    return Image.fromarray(np.array(Image.open(image_path)))


# In[3]:


# x = PlotGrid(figsize=(20, 16))
# image = load_image(img_path)

sharpen_kernel = np.array([
    [-1,-1,-1],
    [-1, 9,-1],
    [-1,-1,-1]
])

low_pass_filter = ImageFilter.GaussianBlur(radius=filter_size)
high_pass_filter = ImageFilter.GaussianBlur(radius=filter_size)
sharpen_filter = ImageFilter.UnsharpMask(radius=filter_size, percent=500, threshold=0)
sharpen_filter = ImageFilter.Kernel(sharpen_kernel.shape, sharpen_kernel.flatten())

# bilateral_filtered_image = np.array(cv2.bilateralFilter(np.array(image), 10, 50, 50))

# x.plot((4, 2, 1), image, title='Color')
# x.plot((4, 2, 2), image.convert('L'), title='Gray')
# x.plot((4, 2, 3), image.filter(low_pass_filter), title='Gaussian Blur (Low Pass Filter)')
# x.plot((4, 2, 4), image.filter(ImageFilter.FIND_EDGES), title='High Pass Filter')
# x.plot((4, 2, 5), image.filter(sharpen_filter), title='Sharpen Image')
# x.plot((4, 2, 6), Image.fromarray(bilateral_filtered_image), title='Bilateral Filter')
# x.plot((4, 2, 7), image.filter(ImageFilter.EDGE_ENHANCE_MORE), title='Test Filter 1')
# x.plot((4, 2, 8), image.filter(ImageFilter.SHARPEN), title='Test Filter 2')


# In[4]:


L5 = np.array([1, 4, 6, 4, 1])
E5 = np.array([-1, -2, 0, 2, 1])
S5 = np.array([-1, 0, 2, 0, -1])
R5 = np.array([1, -4, 6, -4, 1])

L5E5 = np.outer(L5.transpose(), E5)
E5L5 = np.outer(E5.transpose(), L5)

L5R5 = np.outer(L5.transpose(), R5)
R5L5 = np.outer(R5.transpose(), L5)

E5S5 = np.outer(E5.transpose(), S5)
S5E5 = np.outer(S5.transpose(), E5)

S5S5 = np.outer(S5.transpose(), S5)

R5R5 = np.outer(R5.transpose(), R5)

E5E5 = np.outer(E5.transpose(), E5)

L5S5 = np.outer(L5.transpose(), S5)
S5L5 = np.outer(S5.transpose(), L5)

E5R5 = np.outer(E5.transpose(), R5)
R5E5 = np.outer(R5.transpose(), E5)

S5R5 = np.outer(S5.transpose(), R5)
R5S5 = np.outer(R5.transpose(), S5)

filter_sets = {
    'L5E5': [L5E5, E5L5],
    'L5R5': [L5R5, R5L5],
    'E5S5': [E5S5, S5E5],
    'S5S5': [S5S5],
    'R5R5': [R5R5],
    'E5E5': [E5E5],
    'L5S5': [L5S5, S5L5],
    'E5R5': [E5R5, R5E5],
    'S5R5': [S5R5, R5S5]
}


# In[5]:


def norm(np_image):
    for i in range(len(np_image.shape)):
        np_image[:, :, i] = np_image[:, :, i] - np_image[:, :, i].mean()
    return np_image


# In[6]:


def bilateral(image):
    return np.array(cv2.bilateralFilter(np.array(image), 10, 50, 50))

def blur(image):
    return np.array(image.filter(low_pass_filter))

def get_energy(image, image_filter):
    output = signal.convolve(image, image_filter[..., np.newaxis], 'valid')
    return output.sum()

image_transforms = {
    'nonstylized': np.array,
    'blur': blur,
    'bilateral': bilateral
}

def calculate_energy(image_path):
    image = load_image(image_path)
    image_set = [norm(transform(image)) for key, transform in image_transforms.items()]
    energy_set = []
    for image in image_set:
        image_energy = []
        for filter_set_key, filters in filter_sets.items():
            energy = []
            for image_filter in filters:
#                 image_filter = np.repeat(image_filter[:, :, np.newaxis], 3, axis=2)
                energy.append(get_energy(image, image_filter))
            image_energy.append(np.mean(energy))
        energy_set.append(image_energy)
    return energy_set


# In[7]:


def bilateral(image):
    return np.array(cv2.bilateralFilter(np.array(image), 10, 50, 50))

def blur(image):
    return np.array(image.filter(low_pass_filter))

def get_energy(image, image_filter, absolute=False):
    output = signal.convolve(image, image_filter[..., np.newaxis], 'valid')
    if absolute:
        output = abs(output)
    return output.sum()

image_transforms = {
    'raw': np.array,
    'blur': blur,
    'bilateral': bilateral
}

def calculate_energy(image):
    # image = load_image(image_path)
    image_set = [norm(transform(image)) for key, transform in image_transforms.items()]
    energy_set = []
    for image in image_set:
        image_energy = []
        for filter_set_key, filters in filter_sets.items():
            energy = []
            for image_filter in filters:
#                 image_filter = np.repeat(image_filter[:, :, np.newaxis], 3, axis=2)
                energy.append(get_energy(image, image_filter))
            image_energy.append(np.mean(energy))
        energy_set.append(image_energy)
    return energy_set


energy_set_labels = [
    'dataset_index',
    'dataset_class',
    'transformation',
    'L5E5', 'L5R5', 'E5S5',
    'S5S5', 'R5R5', 'E5E5',
    'L5S5', 'E5R5', 'S5R5'
]

def calculate_energy(image, datapoint_index, datapoint_class):
    # image = load_image(image_path)
    image_set = [norm(transform(image)) for key, transform in image_transforms.items()]
    image_keys = [key for key, transform in image_transforms.items()]
    energy_set = pd.DataFrame(columns=energy_set_labels)
    for index, image in enumerate(image_set):
        image_energy = {
            'dataset_index': datapoint_index,
            'dataset_class': datapoint_class
        }
        image_absolute_energy = {
            'dataset_index': datapoint_index,
            'dataset_class': datapoint_class
        }
        for filter_set_key, filters in filter_sets.items():
            energy = []
            absolute_energy = []
            for image_filter in filters:
                energy.append(get_energy(image, image_filter))
                absolute_energy.append(get_energy(image, image_filter, absolute=True))
            image_energy[filter_set_key] = np.mean(energy)
            image_absolute_energy[filter_set_key] = np.mean(absolute_energy)
        image_energy['transformation'] = '{}-actual'.format(image_keys[index])
        image_absolute_energy['transformation'] = '{}-absolute'.format(image_keys[index])
        energy_set = energy_set.append(image_energy, ignore_index=True)
        energy_set = energy_set.append(image_absolute_energy, ignore_index=True)
    return energy_set


# In[8]:


# energy_sets = calculate_energy(img_path)


# In[9]:


def write(energy_sets, name):
    for energy_set_index, image_transform_name in enumerate(image_transforms):
        # print('{:10s} {}'.format(image_transform_name, energy_sets[energy_set_index]))
        log_file_name = '{}-{}.csv'.format(name, image_transform_name)
        log_file_path = os.path.join('csv', log_file_name)
        # log_file = open(log_file_path, 'a')
        # log_file.write('{}\n'.format(', '.join(map(str, energy_sets[energy_set_index]))))
        # log_file.close()
        energy_sets.to_csv(log_file_path, mode='a', header=False)

# display(energy_sets)

for dataset_index, dataset in enumerate([nonstylized_val_dataset, stylized_val_dataset]):
    for datapoint_index, datapoint in enumerate(tqdm(dataset)):
        index_image = dataset.INDEX_IMAGE
        index_target = dataset.INDEX_TARGET
        image = toImage(datapoint[index_image])
        energy_set = calculate_energy(image, datapoint_index, datapoint[index_target])
        # print(energy_set.to_string())
        write(energy_set, 'nonstylized' if dataset_index == 0 else 'stylized')
