import torch
import torch.nn as nn
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.models.inception import BasicConv2d
from torchvision.transforms import transforms


def get_model(dropout, pre_weights=None):
    model = inception_v3(
        weights=Inception_V3_Weights.DEFAULT, dropout=dropout, transform_input=False
    )

    # create new first conv layer (inception)
    weights = model.Conv2d_1a_3x3.conv.weight.clone()
    model.Conv2d_1a_3x3 = nn.Sequential(
        transforms.Resize(299, antialias=True),
        BasicConv2d(5, 32, kernel_size=3, stride=2),
    )

    with torch.no_grad():
        model.Conv2d_1a_3x3[1].conv.weight[:, 1] = weights[:, 0]
        model.Conv2d_1a_3x3[1].conv.weight[:, 1] = weights[:, 0]
        model.Conv2d_1a_3x3[1].conv.weight[:, 2] = weights[:, 0]
        model.Conv2d_1a_3x3[1].conv.weight[:, 3] = weights[:, 0]
        model.Conv2d_1a_3x3[1].conv.weight[:, 4] = weights[:, 0]

    # create new final linear layer
    fully_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(fully_features, 1), nn.Sigmoid())

    if pre_weights:
        model.load_state_dict(torch.load(pre_weights))

    for index, parameter in enumerate(model.parameters()):
        if index < 140:
            parameter.requires_grad = False

    model.float()

    return model
