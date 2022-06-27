import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import albumentations as A


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes, feature_extract, use_pretrained=True):
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft


if __name__ == "__main__":
    num_classes = 2
    feature_extract = True
    model_ft = initialize_model(
        num_classes, 
        feature_extract, 
        use_pretrained=True
    )
    print(model_ft)