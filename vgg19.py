import torch
import torchvision
import torchvision.models as models
from instancenormbatchswap import InstanceNormBatchSwap, InstanceNormSimilarity
from utils import init_weights, pathJoin, convert_input
import os
from betavae import BetaVAE_H


def create_imagenet200_classifier(in_feature_size=25088):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=in_feature_size, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=4096, out_features=200, bias=True)
    )


class VGG_IN(torch.nn.Module):
    def __init__(self, layer_index, instance_normalization_function=None, affine=False, pretrained=False, filename=None):
        super(VGG_IN, self).__init__()
        vgg19 = models.vgg19(pretrained=pretrained)
        self.features1 = vgg19.features[:layer_index]
        if instance_normalization_function is not None:
            if filename:
                self.instance_normalization = instance_normalization_function(
                    vgg19.features[layer_index].out_channels, affine=affine, filename=filename)
            else:
                self.instance_normalization = instance_normalization_function(
                    vgg19.features[layer_index].out_channels, affine=affine)
        self.features2 = vgg19.features[layer_index:]
        self.classifier = create_imagenet200_classifier()

    def forward(self, x):
        x = self.features1(x)
        if hasattr(self, 'instance_normalization'):
            x = self.instance_normalization(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG_COSINE_SIMILARITY(torch.nn.Module):
    def __init__(self, layer_indices=[1, 6, 11, 20, 29], pretrained=False, eps=torch.tensor(1e-08)):
        super(VGG_COSINE_SIMILARITY, self).__init__()
        self.vgg19 = models.vgg19(pretrained=pretrained)
        self.classifier = create_imagenet200_classifier()
        self.layer_indices = layer_indices
        self.eps = eps

    def calculate_similarity_score(self, x):
        # torch.set_printoptions(profile="full")
        flat_x = x.view(x.size(0), x.size(1), -1)
        similarity_matrix = self.calculate_cosine_similarity_matrix(flat_x)
        similarity_matrix = similarity_matrix ** 2
        # print('similarity_matrix.shape')
        # print(similarity_matrix.shape)
        # print(similarity_matrix)
        similarity_score = similarity_matrix.sum(dim=1).sum(dim=1) - similarity_matrix.size(1)
        # print('similarity_score.shape')
        # print(similarity_score.shape)
        # print(similarity_score)
        # torch.set_printoptions(profile="default")
        # raise NotImplemented
        return similarity_score

    def calculate_cosine_similarity_matrix(self, x):
        # print('x.shape')
        # print(x.shape)
        # https://pytorch.org/docs/stable/nn.html#cosine_similarity
        x_t = x.transpose(1, 2)
        # print('x_t.shape')
        # print(x_t.shape)
        x_norm = torch.norm(x, dim=2, keepdim=True)
        # print('x_norm.shape')
        # print(x_norm.shape)
        x_t_norm = x_norm.transpose(1, 2)
        # print('x_t_norm.shape')
        # print(x_t_norm.shape)
        num = torch.matmul(x, x_t)
        # print('num.shape')
        # print(num.shape)
        norm_prod = torch.matmul(x_norm, x_t_norm)
        # print('norm_prod.shape')
        # print(norm_prod.shape)
        den = torch.max(norm_prod, self.eps.to(norm_prod.device))
        # print('den.shape')
        # print(den.shape)
        return num / den


class VGG_IN_SINGLE_SIMILARITY(VGG_COSINE_SIMILARITY):
    def __init__(self, layer_index, instance_normalization_function=None,
                    affine=False, pretrained=False, eps=torch.tensor(1e-08), layer_indices=[1, 6, 11, 20, 29]):
        super(VGG_IN_SINGLE_SIMILARITY, self).__init__(pretrained=pretrained, eps=eps, layer_indices=layer_indices)
        self.features1 = self.vgg19.features[:layer_index]
        if instance_normalization_function is not None:
            self.instance_normalization = instance_normalization_function(
                self.vgg19.features[layer_index].out_channels, affine=affine)
        self.features2 = self.vgg19.features[layer_index:]

    def forward(self, x):
        similarity_scores = []

        current_layer = 0

        # x = self.features1(x)
        for layer_index, layer in enumerate(self.features1):
            x = layer(x)
            if (current_layer in self.layer_indices):
                similarity_scores.append(self.calculate_similarity_score(x))
            current_layer += 1

        if hasattr(self, 'instance_normalization'):
            x = self.instance_normalization(x)

        # x = self.features2(x)
        for layer_index, layer in enumerate(self.features2):
            x = layer(x)
            if (current_layer in self.layer_indices):
                similarity_scores.append(self.calculate_similarity_score(x))
            current_layer += 1

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        layer_similarity = torch.stack(similarity_scores, dim=1)

        return x, layer_similarity


class VGG_BN_SIMILARITY(VGG_COSINE_SIMILARITY):
    def __init__(self, pretrained=False, eps=torch.tensor(1e-08), layer_indices=[2, 9, 16, 29, 42]):
        super(VGG_BN_SIMILARITY, self).__init__(pretrained=pretrained, eps=eps, layer_indices=layer_indices)
        self.vgg19_bn = models.vgg19_bn(pretrained=pretrained)
        self.features = self.vgg19_bn.features

    def forward(self, x):
        similarity_scores = []

        for layer_index, layer in enumerate(self.features):
            x = layer(x)
            if (layer_index in self.layer_indices):
                similarity_scores.append(self.calculate_similarity_score(x))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        layer_similarity = torch.stack(similarity_scores, dim=1)
        return x, layer_similarity


class VGG_VANILLA_SIMILARITY(VGG_COSINE_SIMILARITY):
    def __init__(self, pretrained=False, eps=torch.tensor(1e-08), layer_indices=[1, 6, 11, 20, 29]):
        super(VGG_VANILLA_SIMILARITY, self).__init__(pretrained=pretrained, eps=eps, layer_indices=layer_indices)
        self.features = self.vgg19.features

    def forward(self, x):
        similarity_scores = []

        for layer_index, layer in enumerate(self.features):
            x = layer(x)
            if (layer_index in self.layer_indices):
                similarity_scores.append(self.calculate_similarity_score(x))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        layer_similarity = torch.stack(similarity_scores, dim=1)
        return x, layer_similarity


# Vanilla Models

def create_vgg19_vanilla_scratch():
    # load model from pytorch
    vgg19 = models.vgg19(pretrained=False)

    vgg19.classifier = create_imagenet200_classifier()

    vgg19.apply(init_weights)

    # train all layers
    for param in vgg19.parameters():
        param.requires_grad = True

    return vgg19


def create_vgg19_vanilla_tune_fc():
    # load model from pytorch
    vgg19 = models.vgg19(pretrained=True)

    # freeze cnn layers
    for param in vgg19.parameters():
        param.requires_grad = False

    vgg19.classifier = create_imagenet200_classifier()

    # train fc layers
    for param in vgg19.classifier.parameters():
        param.requires_grad = True

    return vgg19


# Batch Normalization Models

def create_vgg19_bn_all_tune_fc():
    vgg19 = torchvision.models.vgg19_bn(pretrained=True)

    # freeze cnn layers
    for param in vgg19.features.parameters():
        param.requires_grad = False

    vgg19.classifier = create_imagenet200_classifier()

    # train fc layers
    for param in vgg19.classifier.parameters():
        param.requires_grad = True

    return vgg19


def create_vgg19_bn_all_tune_all():
    vgg19 = torchvision.models.vgg19_bn(pretrained=True)

    # train cnn layers
    for param in vgg19.features.parameters():
        param.requires_grad = True

    vgg19.classifier = create_imagenet200_classifier()

    # train fc layers
    for param in vgg19.classifier.parameters():
        param.requires_grad = True

    return vgg19


# Instance Normalization Models

def create_vgg19_in_single_tune_after():
    vgg = VGG_IN(21, instance_normalization_function=torch.nn.InstanceNorm2d, pretrained=True)

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
    vgg = VGG_IN(21, instance_normalization_function=torch.nn.InstanceNorm2d, pretrained=True)

    # train all layers
    for param in vgg.parameters():
        param.requires_grad = True

    return vgg


def create_vgg19_in_affine_single_tune_all():
    vgg = VGG_IN(21, instance_normalization_function=torch.nn.InstanceNorm2d, affine=True, pretrained=True)

    # train all layers
    for param in vgg.parameters():
        param.requires_grad = True

    return vgg


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

    vgg19_in.classifier = create_imagenet200_classifier()

    # train all layers
    for param in vgg19_in.parameters():
        param.requires_grad = True

    return vgg19_in


def create_vgg19_in_all_tune_all():
    return create_vgg19_in_all_tune_all_root()


# Instance Normalization with Batch Statistics

def create_vgg19_in_sm_single_tune_after():
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


def create_vgg19_in_sm_single_tune_all():
    vgg = VGG_IN(21, instance_normalization_function=InstanceNormBatchSwap, pretrained=True)

    # train all layers
    for param in vgg.parameters():
        param.requires_grad = True

    return vgg


def create_vgg19_in_sm_all_tune_all():
    return create_vgg19_in_all_tune_all_root(instance_normalization_function=InstanceNormBatchSwap)


# BN + IN

def create_vgg19_bn_in_single_tune_all_root(instance_normalization_function=None):
    vgg19_in = torchvision.models.vgg19_bn(pretrained=False)
    vgg19 = torchvision.models.vgg19(pretrained=True)
    replace_layers = [28]

    transfer_layer_index = 0
    for layer_index, layer in enumerate(vgg19_in.features):
        # replace batch norm with instance norm
        if instance_normalization_function is not None and \
            isinstance(layer, torch.nn.BatchNorm2d) and \
            layer_index in replace_layers:
            vgg19_in.features[layer_index] = instance_normalization_function(layer.num_features)
    
        # transfer conv weights
        if isinstance(layer, torch.nn.Conv2d):
            vgg19_in.features[layer_index].load_state_dict(vgg19.features[transfer_layer_index].state_dict())

        if (isinstance(layer, torch.nn.ReLU) or
            isinstance(layer, torch.nn.MaxPool2d) or
            isinstance(layer, torch.nn.Conv2d)):
            transfer_layer_index += 1

    vgg19_in.classifier = create_imagenet200_classifier()

    # train all layers
    for param in vgg19_in.parameters():
        param.requires_grad = True

    return vgg19_in


def create_vgg19_bn_in_single_tune_all():
    return create_vgg19_bn_in_single_tune_all_root(instance_normalization_function=torch.nn.InstanceNorm2d)


# Cosine Similarity Models

def create_vgg19_vanilla_similarity_tune_all(): # create_vgg19_cosine_tune_all
    # load model from pytorch
    vgg19 = VGG_VANILLA_SIMILARITY(pretrained=True)

    # train all layers
    for param in vgg19.parameters():
        param.requires_grad = True

    return vgg19


def create_vgg19_in_single_similarity_tune_all(): # create_vgg19_in_single_tune_all
    vgg = VGG_IN_SINGLE_SIMILARITY(21, instance_normalization_function=torch.nn.InstanceNorm2d, pretrained=True)

    # train all layers
    for param in vgg.parameters():
        param.requires_grad = True

    return vgg


def create_vgg19_bn_all_similarity_tune_fc():
    vgg19 = VGG_BN_SIMILARITY(pretrained=True)

    # freeze cnn layers
    for param in vgg19.features.parameters():
        param.requires_grad = False

    # train fc layers
    for param in vgg19.classifier.parameters():
        param.requires_grad = True

    return vgg19


def create_vgg19_bn_all_similarity_tune_all():
    vgg19 = VGG_BN_SIMILARITY(pretrained=True)

    # train cnn layers
    for param in vgg19.features.parameters():
        param.requires_grad = True

    # train fc layers
    for param in vgg19.classifier.parameters():
        param.requires_grad = True

    return vgg19


def create_vgg19_in_sm_single_similarity(filename):
    vgg19 = VGG_IN(21, instance_normalization_function=InstanceNormSimilarity, pretrained=True, filename=filename)

    # train all layers
    for param in vgg19.parameters():
        param.requires_grad = False

    return vgg19


class VGG_IN_VAE(torch.nn.Module):
    def __init__(self, classification_model, autoencoder_model, vae_transforms, z_dim=128, feature_dim=25088):
        super(VGG_IN_VAE, self).__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.get_features = self.create_feature_extractor(classification_model)
        self.get_latents = self.create_latent_extractor(autoencoder_model)
        self.classifier = create_imagenet200_classifier(self.feature_dim + self.z_dim)
        self.convert_input = convert_input(vae_transforms)

    def forward(self, x):
        with torch.no_grad():
            features = self.get_features(x)
            latents = self.get_latents(self.convert_input(x))
        combined = torch.cat([features, latents], dim=1)
        output = self.classifier(combined)
        return output
    
    def create_feature_extractor(self, classification_model):
        def feature_extractor(x):
            with torch.no_grad():
                features = classification_model.features1(x)
                if hasattr(classification_model, 'instance_normalization'):
                    features = classification_model.instance_normalization(features)
                features = classification_model.features2(features)
                features = features.view(features.size(0), -1)
            return features
        return feature_extractor
    
    def create_latent_extractor(self, autoencoder_model):
        def latent_extractor(x):
            with torch.no_grad():
                encoded_x = autoencoder_model._encode(x)
                encoded_x = encoded_x[:, :self.z_dim] # take the mean
            return encoded_x
        return latent_extractor

def create_feature_latent_classifier(classification_modelname, autoencoder_modelname, z_dim, model_directory, device, vae_transforms):
    def assemble_model():
        classification_model = create_vgg19_in_single_tune_all()
        # load trained classifier
        classification_model_checkpoint_path = pathJoin(model_directory, '{}.ckpt'.format(classification_modelname))
        if os.path.isfile(classification_model_checkpoint_path):
            classification_model.load_state_dict(
                torch.load(classification_model_checkpoint_path, map_location=device)['weights'])
            classification_model.eval()
        else:
            raise ValueError('Classification Model not found at: {}'.format(classification_model_checkpoint_path))
        # load trained vae
        autoencoder_model_checkpoint_path = pathJoin(model_directory, '{}.ckpt'.format(autoencoder_modelname))
        autoencoder_model = BetaVAE_H(z_dim=z_dim, nc=3)
        if os.path.isfile(autoencoder_model_checkpoint_path):
            autoencoder_model.load_state_dict(
                torch.load(autoencoder_model_checkpoint_path, map_location=device)['weights'])
            autoencoder_model.eval()
        else:
            raise ValueError('Autoencoder Model not found at: {}'.format(autoencoder_model_checkpoint_path))
        twin_model = VGG_IN_VAE(classification_model, autoencoder_model, vae_transforms, z_dim=z_dim)
        return twin_model
    return assemble_model


class VGG_IN_LATENT(VGG_IN):
    def __init__(self, layer_index, autoencoder_model, vae_transforms,
        instance_normalization_function=None, affine=False,
        pretrained=False, z_dim=128, feature_dim=25088):
        super(VGG_IN_LATENT, self).__init__(layer_index,
            instance_normalization_function=instance_normalization_function, affine=affine, pretrained=pretrained)
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.get_latents = self.create_latent_extractor(autoencoder_model)
        self.classifier = create_imagenet200_classifier(self.feature_dim + self.z_dim)
        self.vae_transforms = vae_transforms

    def forward(self, x):
        x = self.features1(x)
        if hasattr(self, 'instance_normalization'):
            x = self.instance_normalization(x)
        x = self.features2(x)
        features = x.view(x.size(0), -1)
        with torch.no_grad():
            latents = self.get_latents(self.convert_input(x))
        combined = torch.cat([features, latents], dim=1)
        output = self.classifier(combined)
        return output
    
    def create_latent_extractor(self, autoencoder_model):
        def latent_extractor(x):
            with torch.no_grad():
                encoded_x = autoencoder_model._encode(x)
                encoded_x = encoded_x[:, :self.z_dim] # take the mean
            return encoded_x
        return latent_extractor
    
    def convert_input(self, x):
        return torch.stack([ self.vae_transforms(_) for _ in x.cpu() ], dim=0).cuda()


def create_vgg19_in_single_tune_all_with_latent(autoencoder_model_checkpoint_path, z_dim, device, vae_transforms):
    def assemble_model():
        # load trained vae
        autoencoder_model = BetaVAE_H(z_dim=z_dim, nc=3)
        if os.path.isfile(autoencoder_model_checkpoint_path):
            autoencoder_model.load_state_dict(
                torch.load(autoencoder_model_checkpoint_path, map_location=device)['weights'])
            autoencoder_model.eval()
        else:
            raise ValueError('Autoencoder Model not found at: {}'.format(autoencoder_model_checkpoint_path))

        vgg = VGG_IN_LATENT(21, autoencoder_model, vae_transforms, instance_normalization_function=torch.nn.InstanceNorm2d, pretrained=True, z_dim=z_dim)

        # train all layers
        for param in vgg.parameters():
            param.requires_grad = True

        return vgg
    return assemble_model
