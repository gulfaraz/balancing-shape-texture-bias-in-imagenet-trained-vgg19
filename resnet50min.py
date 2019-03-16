import torch
import torchvision

def create_miniimagenet_classifier():
    return torch.nn.Linear(in_features=2048, out_features=200, bias=True)

def create_resnet50_tune_fc(pretrained=None):
    resnet50 = torchvision.models.resnet50(pretrained=pretrained)
    resnet50.fc = create_miniimagenet_classifier()
    return resnet50

