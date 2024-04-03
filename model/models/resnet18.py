import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def get_model(dropout, pre_weights=None):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # create new first conv layer (resnet)
    conv1_weights = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight[:, 0] = conv1_weights[:, 0]
        model.conv1.weight[:, 1] = conv1_weights[:, 0]
        model.conv1.weight[:, 2] = conv1_weights[:, 0]
        model.conv1.weight[:, 3] = conv1_weights[:, 0]
        model.conv1.weight[:, 4] = conv1_weights[:, 0]

    # create new final linear layer
    fully_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout), nn.Linear(fully_features, 1), nn.Sigmoid()
    )

    if pre_weights:
        model.load_state_dict(torch.load(pre_weights))

    for index, parameter in enumerate(model.parameters()):
        if index < 30:
            parameter.requires_grad = False

    model.float()

    return model
