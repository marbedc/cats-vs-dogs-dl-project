import torch.nn as nn
from torchvision import models

def get_pretrained_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
