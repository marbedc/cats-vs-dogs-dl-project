import torch.nn as nn
from torchvision import models


def get_vit_model():
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    # Freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classification head for binary classification
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 1)

    return model
