import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from models.resnet18 import get_model as get_resnet18_model
from models.resnet50 import get_model as get_resnet50_model
from models.inception_v3 import get_model as get_inception_model
from models.efficientnet_s import get_model as get_efficientnet_s_model
from models.efficientnet_m import get_model as get_efficientnet_m_model

from dataset import GeoEra5Dataset, GeoUkesmDataset, SlpEra5Dataset, SlpUkesmDataset

import torch
import albumentations as A


def get_image(data, time):
    fig, axs = plt.subplots(
        nrows=5, ncols=1, subplot_kw={"projection": ccrs.PlateCarree()}
    )

    axs = axs.flatten()
    clevs = np.arange(-5, 5, 1)
    long = np.arange(-45, 55, 1)
    lat = np.arange(30, 75, 1)

    for i in range(5):
        time = time + datetime.timedelta(days=1)
        axs[i].coastlines(resolution="110m", linewidth=1)
        cs = axs[i].contourf(
            long,
            lat,
            data[i].cpu(),
            clevs,
            transform=ccrs.PlateCarree(),
            cmap=plt.cm.jet,
        )
        if i == 0:
            axs[i].set_title(str(time))

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cs, cax=cbar_ax, orientation="vertical")

    plt.draw()

    fig_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_np = fig_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig_np = fig_np.transpose((2, 0, 1))

    plt.close(fig)

    return fig_np


def get_dataset(key):
    if key == "era5":
        return GeoEra5Dataset()
    elif key == "ukesm":
        return GeoUkesmDataset()
    elif key == "era5-msl":
        return SlpEra5Dataset()
    elif key == "ukesm-msl":
        return SlpUkesmDataset()


def get_date(offset, dataset):
    if "era5" in dataset:
        return datetime.datetime(1900, 1, 1) + datetime.timedelta(hours=int(offset))
    if "ukesm" in dataset:
        reference_date = datetime.datetime(1850, 1, 1)
        return reference_date + datetime.timedelta(days=int(offset / 360 * 365.25))


def get_transform(key):
    transforms = {
        "light": A.Compose(
            [
                A.GaussNoise(p=0.1),
                A.Rotate(limit=15, p=0.1),
                A.ChannelDropout(channel_drop_range=(1, 1), p=0.1),
            ]
        ),
        "heavy": A.Compose(
            [
                A.GaussNoise(p=0.2),
                A.Rotate(limit=30, p=0.2),
                A.ChannelDropout(channel_drop_range=(1, 2), p=0.1),
            ]
        ),
        "none": None,
    }
    return transforms[key]


def get_model(key, dropout, weights=None):
    models = {
        "resnet18": get_resnet18_model,
        "resnet50": get_resnet50_model,
        "efficientnet_s": get_efficientnet_s_model,
        "efficientnet_m": get_efficientnet_m_model,
        "inception": get_inception_model,
    }
    return models[key](dropout=dropout, pre_weights=weights)


def get_optimizer(key, weight_decay, lr, model):
    optimizers = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "adagrad": optim.Adagrad,
    }

    if key == "sgd_0":
        optimizer = optimizers["sgd"](
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0
        )
    elif key == "sgd_09":
        optimizer = optimizers["sgd"](
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        optimizer = optimizers[key](
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    return optimizer


def get_scheduler(key, optimizer):
    schedulers = {
        "step_01": lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
        "step_09": lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9),
        "plateau": lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3),
        "none": None,
    }
    return schedulers[key]


def get_train_loader(dataset, sampler, batch_size, sampler_type):
    if sampler_type == "weighted_random":
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, sampler=sampler
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

    return train_loader
