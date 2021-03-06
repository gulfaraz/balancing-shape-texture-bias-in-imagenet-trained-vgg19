
# coding: utf-8

# In[1]: Load Libraries

# native
import os
import sys

# modules
from utils import *
from dataset import *
from vgg19 import *
from betavae import *
from score import *
from trainer import *
from logger import *

# pytorch
import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import cv2

# In[2]: Check Requirements

requirements = {
    torch: '1'
}

check_requirements(requirements)


config = configuration()
for k, v in sorted(vars(config).items()):
    print('{0}: {1}'.format(k, v))

# In[3]: Load Datasets

IMAGE_SIZE = (config.inputSize, config.inputSize)
VAE_IMAGE_SIZE = (config.vaeImageSize, config.vaeImageSize)

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

bilateral_train_transforms = transforms.Compose([
    lambda x: np.array(cv2.bilateralFilter(np.array(x), 10, 100, 50)),
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(IMAGE_SIZE[0]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

bilateral_test_transforms = transforms.Compose([
    lambda x: np.array(cv2.bilateralFilter(np.array(x), 10, 100, 50)),
    transforms.ToPILImage(),
    transforms.Resize(roundUp(IMAGE_SIZE[0])),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    normalize
])

vae_transforms = transforms.Compose([
    transforms.Resize(VAE_IMAGE_SIZE),
    transforms.ToTensor()
])

convert_to_vae_transforms = transforms.Compose([
    denormalize,
    transforms.ToPILImage(),
    transforms.Resize(VAE_IMAGE_SIZE),
    transforms.ToTensor()
])

highpass_transforms = transforms.Compose([
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    lambda x: (x > 0.2).float()
])

def load_data(dataset_name, split, train_transforms=train_transforms, test_transforms=test_transforms):
    dataset_path = os.path.join(config.rootPath, 'datasets', dataset_name)

    istrain = split == 'train'
    transforms = train_transforms if istrain else test_transforms

    dataset = ImageNet200Dataset(dataset_path, split=split, transforms=transforms)#raw_transforms)
    # dataset = CelebADataset('./space/datasets/CelebA/img_align_celeba', transforms=transforms)
    loader = DataLoader(dataset, batch_size=config.batchSize, shuffle=istrain, num_workers=config.numberOfWorkers)

    print('{} dataset {} has {} datapoints in {} batches'.format(split, dataset_name, len(dataset), len(loader)))

    return dataset, loader

def load_bilateral_data(dataset_name, split,
        train_transforms=bilateral_train_transforms, test_transforms=bilateral_test_transforms):
    dataset_path = os.path.join(config.rootPath, 'datasets', dataset_name)

    istrain = split == 'train'
    transforms = train_transforms if istrain else test_transforms

    dataset = ImageNet200Dataset(dataset_path, split=split, transforms=transforms)#raw_transforms)
    # dataset = CelebADataset('./space/datasets/CelebA/img_align_celeba', transforms=transforms)
    loader = DataLoader(dataset, batch_size=config.batchSize, shuffle=istrain, num_workers=config.numberOfWorkers)

    print('{} dataset {} has {} datapoints in {} batches'.format(split, dataset_name, len(dataset), len(loader)))

    return dataset, loader

def load_pair_data(dataset_names, split, target_type):
    input_dataset_path = os.path.join(config.rootPath, 'datasets', dataset_names[0])
    target_dataset_path = os.path.join(config.rootPath, 'datasets', dataset_names[1])

    istrain = split == 'train'
    target_transforms = highpass_transforms if target_type == 'highpass' else vae_transforms

    dataset = ImageNet200PairDataset(input_dataset_path, target_dataset_path, split=split,
        transforms=vae_transforms, target_type=target_type, target_transforms=target_transforms)
    # dataset = CelebADataset('./space/datasets/CelebA/img_align_celeba', transforms=vae_transforms)
    loader = DataLoader(dataset, batch_size=config.batchSize, shuffle=istrain, num_workers=config.numberOfWorkers)

    print('{} dataset pair ({}, {}) has {} datapoints in {} batches'.format(split, dataset_names[0], dataset_names[1],
        len(dataset), len(loader)))

    return dataset, loader

original_train_dataset, original_train_loader = load_data('imagenet200', 'train')
original_val_dataset, original_val_loader = load_data('imagenet200', 'val')

stylized_train_dataset, stylized_train_loader = load_data('stylized-imagenet200-1.0', 'train')
stylized_val_dataset, stylized_val_loader = load_data('stylized-imagenet200-1.0', 'val')

bilateral_original_train_dataset, bilateral_original_train_loader = load_data('imagenet200', 'train',
    train_transforms=bilateral_train_transforms, test_transforms=bilateral_test_transforms)
bilateral_original_val_dataset, bilateral_original_val_loader = load_data('imagenet200', 'val',
    train_transforms=bilateral_train_transforms, test_transforms=bilateral_test_transforms)

_, nonstylized_nonstylized_loader = load_pair_data(['stylized-imagenet200-0.0', 'stylized-imagenet200-1.0'],
                            'train', 'nonstylized')

for dataset, loader in [
    (original_train_dataset, original_train_loader),
    (original_val_dataset, original_val_loader),
    (stylized_train_dataset, stylized_train_loader),
    (stylized_val_dataset, stylized_val_loader)
]:
    print('{} Datapoints in {} Batches'.format(len(dataset), len(loader)))

dataset_names = [
    'stylized-imagenet200-1.0', 'stylized-imagenet200-0.9', 'stylized-imagenet200-0.8',
    'stylized-imagenet200-0.7', 'stylized-imagenet200-0.6', 'stylized-imagenet200-0.5',
    'stylized-imagenet200-0.4', 'stylized-imagenet200-0.3', 'stylized-imagenet200-0.2',
    'stylized-imagenet200-0.1', 'stylized-imagenet200-0.0', 'imagenet200'
]

# In[4]: Setup Models

# models directory
model_directory = pathJoin(config.rootPath, 'models')

vae_model_name = '{}_beta{}_gamma{}'.format(config.zdim, config.beta, config.gamma)
vae_checkpoint_model_name = 'vae{}'.format(vae_model_name)
vae_checkpoint_model_name = 'bilateral_{}'.format(vae_checkpoint_model_name) if config.bilateral else vae_checkpoint_model_name
vae_checkpoint_model_name = '{}_{}'.format(config.dataset, vae_checkpoint_model_name)
vae_model_checkpoint_path = pathJoin(model_directory, '{}.ckpt'.format(vae_checkpoint_model_name))

supported_models = {
    # baseline
    'vgg19_vanilla_tune_fc': create_vgg19_vanilla_tune_fc, # Vanilla (No Norm)
    # normalization
    'vgg19_bn_all_tune_fc': create_vgg19_bn_all_tune_fc, # Batch Norm
    'vgg19_bn_in_single_tune_all': create_vgg19_bn_in_single_tune_all, # Batch Norm with Single IN
    'vgg19_in_all_tune_all': create_vgg19_in_all_tune_all, # Instance Norm
    'vgg19_in_single_tune_all': create_vgg19_in_single_tune_all, # Single IN
    'vgg19_in_affine_single_tune_all': create_vgg19_in_affine_single_tune_all, # Single IN with Affine
    'vgg19_in_sm_all_tune_all': create_vgg19_in_sm_all_tune_all, # IN-SM
    'vgg19_in_sm_single_tune_all': create_vgg19_in_sm_all_tune_all, # Single IN-SM
    # similarity
    'similarity_vgg19_vanilla_tune_all': create_vgg19_vanilla_similarity_tune_all,
    'similarity_vgg19_in_single_tune_all': create_vgg19_in_single_similarity_tune_all,
    'similarity_vgg19_bn_all_tune_fc': create_vgg19_bn_all_similarity_tune_fc,
    # latent representation
    'vae{}'.format(vae_model_name): create_betavae(config.zdim),
    'classifier_z{}'.format(vae_model_name): create_betavae_classifier(
        vae_model_checkpoint_path, config.zdim, config.device),
    # train with latent
    'latent_vgg19_in_single_tune_all': create_vgg19_in_single_tune_all_with_latent(
        vae_model_checkpoint_path, config.zdim, config.device, convert_to_vae_transforms), # Single IN with Latent
}

selected_models = {}

for model_name, model_constructor in supported_models.items():
    selected_model_name = model_name
    if config.dataset != 'nonstylized' and \
        not (
            model_name == 'vgg19_vanilla_tune_fc' or \
            'vae' in model_name or \
            'classifier' in model_name or \
            'latent' in model_name
        ):
        continue
    if config.bilateral:
        selected_model_name = 'bilateral_{}'.format(model_name)
        if 'similarity' in model_name or \
            'vae' in model_name or \
            'classifier' in model_name or \
            'latent' in model_name:
            continue
    selected_model_name = '{}_{}'.format(config.dataset, selected_model_name)
    selected_models[selected_model_name] = model_constructor

supported_models = selected_models

# In[5]: Sanity Check

models = {k:v for (k,v) in supported_models.items()
            if k in (config.model if config.model is not None else supported_models)}
assert len(models.keys()) > 0, 'Please specify a model'

sanity(models, original_train_loader, nonstylized_nonstylized_loader, config.device)

# In[6]: Train Models

if config.train:

    similarity_weight = 0.04

    # setup log directory
    log_directory = pathJoin('run_logs')
    os.makedirs(log_directory, exist_ok=True)

    for model_name in models:
        logger = create_logger(log_directory, model_name)
        logger.info(' '.join(sys.argv))
        logger.info('Model Name {}'.format(model_name))
        model = models[model_name]()
        if 'vae' in model_name:
            target_type = model_name.split('_')[0]
            _, pair_train_loader = load_pair_data(['stylized-imagenet200-0.0', 'stylized-imagenet200-1.0'],
                                        'train', target_type)
            _, pair_val_loader = load_pair_data(['stylized-imagenet200-0.0', 'stylized-imagenet200-1.0'],
                                        'val', target_type)
            run_autoencoder(
                model_name,
                model,
                model_directory,
                config.numberOfEpochs,
                config.autoencoderLearningRate,
                logger,
                pair_train_loader,
                pair_val_loader,
                config.device,
                config.beta,
                config.vaeImageSize,
                config.gamma,
                load_data=load_data,
                vae_transforms=convert_to_vae_transforms
            )
        else:
            train_loader = original_train_loader
            val_loader = original_val_loader
            if 'bilateral' in model_name:
                train_loader = bilateral_original_train_loader
                val_loader = bilateral_original_val_loader
            elif config.dataset == 'stylized':
                train_loader = stylized_train_loader
                val_loader = stylized_val_loader
            run(
                model_name, model,
                model_directory,
                config.numberOfEpochs,
                config.learningRate,
                logger,
                train_loader,
                val_loader,
                config.device,
                similarity_weight=similarity_weight if 'similarity' in model_name else None,
                load_data=load_data
            )

        del model
        torch.cuda.empty_cache()

# In[6]: Check Performance

perf(models, model_directory, dataset_names, config.device, load_data=load_data, load_bilateral_data=load_bilateral_data, only_exists=config.exists, vae_transforms=convert_to_vae_transforms)

