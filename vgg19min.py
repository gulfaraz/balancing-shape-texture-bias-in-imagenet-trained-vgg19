import torch
import torchvision.models as models
from instancenormbatchswap import InstanceNormBatchSwap

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


class VGG_IN_(torch.nn.Module):
    def __init__(self, layer_index, pretrained=False):
        super(VGG_IN_, self).__init__()
        vgg19 = models.vgg19(pretrained=pretrained)
        self.features1 = vgg19.features[:layer_index]
        self.features2 = vgg19.features[layer_index:]
        self.classifier = create_miniimagenet_classifier()

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


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


def create_vgg19_bn_single_in_tune_all_root(instance_normalization_function=None):
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


def create_vgg19_bn_single_in_tune_all():
    return create_vgg19_in_all_tune_all_root(instance_normalization_function=torch.nn.InstanceNorm2d)

