import torch
from resnet_custom import *
from batchinstancenorm import BatchInstanceNorm2d

def create_miniimagenet_classifier():
    return torch.nn.Linear(in_features=2048, out_features=200, bias=True)

def create_resnet50_bn_tune_fc(pretrained=None):
    model = resnet50(pretrained=pretrained)
    model.fc = create_miniimagenet_classifier()
    return model

def create_resnet50_in_tune_fc(pretrained=None):
    model = resnet50(pretrained=pretrained, norm_layer=torch.nn.InstanceNorm2d)
    model.fc = create_miniimagenet_classifier()
    return model

def create_resnet50_bin_tune_fc(pretrained=None):
    model = resnet50(pretrained=pretrained, norm_layer=BatchInstanceNorm2d)
    model.fc = create_miniimagenet_classifier()
    return model
