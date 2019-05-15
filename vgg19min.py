import torch
import torchvision
import torchvision.models as models
from instancenormbatchswap import InstanceNormBatchSwap, InstanceNormSimilarity
from utils import init_weights, pathJoin
import os


def create_miniimagenet_classifier(dim_multiplier=1):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=(25088 * dim_multiplier), out_features=4096, bias=True),
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
        self.classifier = create_miniimagenet_classifier()

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
        self.classifier = create_miniimagenet_classifier()
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

    vgg19.classifier = create_miniimagenet_classifier()

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

    vgg19.classifier = create_miniimagenet_classifier()

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

    vgg19_in.classifier = create_miniimagenet_classifier()

    # train all layers
    for param in vgg19_in.parameters():
        param.requires_grad = True

    return vgg19_in


def create_vgg19_in_all_tune_all():
    return create_vgg19_in_all_tune_all_root()


# Instance Normalization with Batch Statistics

def create_vgg19_in_bs_single_tune_after():
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


def create_vgg19_in_bs_eval():
    vgg = VGG_IN(21, pretrained=True)

    # freeze all layers
    for param in vgg.parameters():
        param.requires_grad = False

    return vgg


def create_vgg19_in_bs_all_tune_all():
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

    vgg19_in.classifier = create_miniimagenet_classifier()

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


def create_vgg19_in_bs_single_similarity(filename):
    vgg19 = VGG_IN(21, instance_normalization_function=InstanceNormSimilarity, pretrained=True, filename=filename)

    # train all layers
    for param in vgg19.parameters():
        param.requires_grad = False

    return vgg19


class VGG_AUTOENCODER(torch.nn.Module):
    def __init__(self, pretrained=False):
        super(VGG_AUTOENCODER, self).__init__()
        vgg19 = models.vgg19(pretrained=pretrained)
        self.encoder = torch.nn.ModuleList([])
        self.decoder = torch.nn.ModuleList([])
        for layer in vgg19.features:
            self.encoder.append(self.get_encoding_layer(layer))
            self.decoder.insert(0, self.get_decoding_layer(layer))
        self.classifier = create_miniimagenet_classifier()

    def forward(self, x, classify=True):
        indices = []
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, torch.nn.MaxPool2d):
                indices.append(x[1])
                x = x[0]
        if classify:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        else:
            for layer in self.decoder:
                if isinstance(layer, torch.nn.MaxUnpool2d):
                    x = layer(x, indices.pop())
                else:
                    x = layer(x)
        return x
    
    def set_mode(self, mode):
        for params in self.encoder.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        for params in self.decoder.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        for params in self.classifier.parameters():
            params.requires_grad = (mode == 'train-classifier')

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

def create_vgg19_autoencoder():
    autoencoder = VGG_AUTOENCODER(pretrained=False)
    autoencoder.set_mode('eval')
    return autoencoder


class VGG_VAE(torch.nn.Module):
    def __init__(self, pretrained=False, num_classes=200, latent_size=4096):
        super(VGG_VAE, self).__init__()
        vgg19 = models.vgg19(pretrained=pretrained)
        self.encoder = torch.nn.ModuleList([])
        self.decoder = torch.nn.ModuleList([])
        for layer in vgg19.features:
            self.encoder.append(self.get_encoding_layer(layer))
            self.decoder.insert(0, self.get_decoding_layer(layer))

        feature_size = 25088
        self.z_size = latent_size

        self.classifier = torch.nn.Linear(in_features=self.z_size, out_features=num_classes, bias=True)

        self.encoder_to_latent = torch.nn.Sequential(
            torch.nn.Linear(feature_size, 10240),
            torch.nn.BatchNorm1d(10240),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(10240, self.z_size),
            torch.nn.BatchNorm1d(self.z_size),
            torch.nn.ReLU(inplace=True)
        )

        self.latent_to_mu = torch.nn.Linear(self.z_size, self.z_size)
        self.latent_to_logvar = torch.nn.Linear(self.z_size, self.z_size)

        self.latent_to_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.z_size, self.z_size),
            torch.nn.BatchNorm1d(self.z_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.z_size, 10240),
            torch.nn.BatchNorm1d(10240),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(10240, feature_size),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(inplace=True)
        )
    
    def encode(self, x):
        indices = []
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, torch.nn.MaxPool2d):
                indices.append(x[1])
                x = x[0]
        features_shape = x.shape
        x = x.view(x.size(0), -1)
        x = self.encoder_to_latent(x)
        return x, indices, features_shape
    
    def decode(self, z, indices=None, feature_shape=None):
        x = self.latent_to_decoder(z)
        x = x.view(*feature_shape)
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

    def forward(self, x, classify=True):
        x, indices, feature_shape = self.encode(x)
        if classify:
            x = self.classifier(x)
            return x
        else:
            mu = self.latent_to_mu(x)
            logvar = self.latent_to_logvar(x)
            z = self.reparameterize(mu, logvar)
            x = self.decode(z, indices=indices, feature_shape=feature_shape)
            return x, mu, logvar
    
    def set_mode(self, mode):
        for params in self.encoder.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        for params in self.encoder_to_latent.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        for params in self.latent_to_mu.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        for params in self.latent_to_logvar.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        for params in self.latent_to_decoder.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        for params in self.decoder.parameters():
            params.requires_grad = (mode == 'train-autoencoder')
        for params in self.classifier.parameters():
            params.requires_grad = (mode == 'train-classifier')

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

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.Tensor(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

def create_vgg19_variational_autoencoder():
    vae = VGG_VAE(pretrained=False)
    vae.set_mode('eval')
    return vae


class VGG_IN_VAE(torch.nn.Module):
    def __init__(self, classification_model, autoencoder_model):
        super(VGG_IN_VAE, self).__init__()
        self.classification_network = self.classification_feature_extractor(classification_model)
        # # disable params
        # for name, param in self.classification_model.features1.named_parameters():
        #     param.requires_grad = False
        # if hasattr(self.classification_model, 'instance_normalization'):
        #     for name, param in self.classification_model.instance_normalization.named_parameters():
        #         param.requires_grad = False
        # for name, param in self.classification_model.features2.named_parameters():
        #     param.requires_grad = False
        # self.classification_model.eval()
        # self.autoencoder_model = autoencoder_model
        # self.autoencoder_model.eval()
        # self.autoencoder_model.set_mode('eval')
        self.autoencoder_network = self.autoencoder_feature_extractor(autoencoder_model)
        self.classifier = create_miniimagenet_classifier(2)

    def forward(self, x):
        with torch.no_grad():
            classification_features = self.classification_network(x)
            print(classification_features.shape)
            autoencoder_features = self.autoencoder_network(x)
            print(autoencoder_features.shape)

        # print('classifier params')
        classifier_input = torch.cat([classification_features, autoencoder_features], dim=1)
        # for name, param in self.classifier.named_parameters():
        #     print(name, param.requires_grad)
        # print(classifier_input.shape)
        output = self.classifier(classifier_input)
        # print(output.shape)

        return output
    
    def classification_feature_extractor(self, classification_model):
        def feature_extractor(x):
            # print('classification params')
            # for name, param in self.classification_model.features1.named_parameters():
            #     print(name, param.requires_grad)
            classification_features = classification_model.features1(x)
            if hasattr(classification_model, 'instance_normalization'):
                classification_features = classification_model.instance_normalization(classification_features)
                # for name, param in self.classification_model.instance_normalization.named_parameters():
                #     print(name, param.requires_grad)
            classification_features = classification_model.features2(classification_features)
            # for name, param in self.classification_model.features2.named_parameters():
            #     print(name, param.requires_grad)
            classification_features = classification_features.view(classification_features.size(0), -1)
            # print(classification_features.shape)
            return classification_features
        return feature_extractor
    
    def autoencoder_feature_extractor(self, autoencoder_model):
        def feature_extractor(x):
            # print('autoencoder params')
            autoencoder_features, _ = autoencoder_model.encode(x)
            # for name, param in self.autoencoder_model.encoder.named_parameters():
            #     print(name, param.requires_grad)
            autoencoder_features = autoencoder_features.view(autoencoder_features.size(0), -1)
            # print(autoencoder_features.shape)
            return autoencoder_features
        return feature_extractor

def create_vgg19_vae_support(classification_modelname, autoencoder_modelname, model_directory, device):
    def assemble_model():
        classification_model = create_vgg19_in_single_tune_all()
        autoencoder_model = create_vgg19_variational_autoencoder()
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
        if os.path.isfile(autoencoder_model_checkpoint_path):
            autoencoder_model.load_state_dict(
                torch.load(autoencoder_model_checkpoint_path, map_location=device)['weights'])
            autoencoder_model.eval()
        else:
            raise ValueError('Autoencoder Model not found at: {}'.format(autoencoder_model_checkpoint_path))
        twin_model = VGG_IN_VAE(classification_model, autoencoder_model)
        print(twin_model)
        return twin_model
    return assemble_model
