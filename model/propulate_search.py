import random
import torch
import logging


from mpi4py import MPI

import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from dataset import (
    TransformDataset,
)

from propulate.utils import get_default_propagator, set_logger_config
from propulate.propulator import Propulator


import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from torchvision.models.inception import Inception3
from util import (
    get_dataset,
    get_date,
    get_optimizer,
    get_scheduler,
    get_model,
    get_transform,
    get_train_loader,
)


def ind_loss(params):
    rank = MPI.COMM_WORLD.rank
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    f1 = BinaryF1Score(threshold=0.5).to(device)
    recall = BinaryRecall(threshold=0.5).to(device)
    precision = BinaryPrecision(threshold=0.5).to(device)

    mean_f1 = np.zeros(NUM_EPOCHS)
    mean_loss = np.zeros(NUM_EPOCHS)
    mean_precision = np.zeros(NUM_EPOCHS)
    mean_recall = np.zeros(NUM_EPOCHS)

    if DEBUG:
        validation_writer = SummaryWriter(
            f"{TENSORBOARD_PREFIX}/val/{VARIABLE}/{params['model']}/{params['lr']}/{params['batch_size']}/{params['sampler']}/{params['loss']}/{params['optimizer']}/{params['scheduler']}/{params['dropout']:.2f}/{params['weight_decay']:.2f}"
        )
        training_writer = SummaryWriter(
            f"{TENSORBOARD_PREFIX}/trn/{VARIABLE}/{params['model']}/{params['lr']}/{params['batch_size']}/{params['sampler']}/{params['loss']}/{params['optimizer']}/{params['scheduler']}/{params['dropout']:.2f}/{params['weight_decay']:.2f}"
        )

    kf = KFold(n_splits=NUM_FOLDS, shuffle=False)

    count_folds = 0
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
        count_folds += 1

        model = get_model(params["model"], params["dropout"])
        model.to(device)

        optimizer = get_optimizer(
            params["optimizer"], params["weight_decay"], params["lr"], model
        )

        scheduler = get_scheduler(params["scheduler"], optimizer)

        train_ds = Subset(train_dataset, train_indices)

        subset_data = [train_dataset[idx] for idx in train_ds.indices]
        _, subset_labels, _ = zip(*subset_data)
        labels = torch.tensor(subset_labels).long()
        train_counts = torch.bincount(labels)
        train_class_weights = len(labels) / (2.0 * train_counts.float())
        train_weights = train_class_weights[labels]
        train_sampler = WeightedRandomSampler(train_weights, len(labels))

        train_ds = TransformDataset(
            subset_data, transform=get_transform(params["augmentation"])
        )

        train_loader = get_train_loader(
            train_ds, train_sampler, params["batch_size"], params["sampler"]
        )

        val_loader = torch.utils.data.DataLoader(
            Subset(train_dataset, val_indices),
            batch_size=params["batch_size"],
            shuffle=False,
        )

        for epoch in range(NUM_EPOCHS):
            ### TRAINING ###

            model.train()
            epoch_loss = 0.0
            epoch_labels = torch.tensor([])
            epoch_outputs = torch.tensor([])
            for inputs, labels, _ in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # fix for inception model in pytorch
                # https://discuss.pytorch.org/t/inception-v3-is-not-working-very-well/38296/3
                if type(model) is Inception3:
                    outputs = model(inputs.float())[0]
                else:
                    outputs = model(inputs.float())

                if params["loss"] == "bce":
                    criterion = nn.BCELoss()
                elif params["loss"] == "bce_weighted":
                    # scale loss weights by class imbalance in input data
                    class_counts = torch.bincount(labels.long())
                    class_weights = params["batch_size"] / (2.0 * class_counts.float())
                    sample_weights = class_weights[labels.long()]
                    criterion = nn.BCELoss(weight=sample_weights)

                loss = criterion(outputs.flatten(), labels.float())
                loss.backward()
                optimizer.step()

                epoch_loss += outputs.shape[0] * loss.item()
                epoch_labels = torch.cat(
                    (epoch_labels, labels.float().detach().cpu()), 0
                )
                epoch_outputs = torch.cat(
                    (epoch_outputs, outputs.flatten().detach().cpu()), 0
                )

            epoch_loss = epoch_loss / len(epoch_labels)
            epoch_predictions = (epoch_outputs > 0.5).float()

            if DEBUG and fold == 0:
                training_writer.add_scalar("loss", epoch_loss, epoch)
                training_writer.add_scalar("f1", f1(epoch_outputs, epoch_labels), epoch)
                training_writer.add_scalar(
                    "recall", recall(epoch_outputs, epoch_labels), epoch
                )
                training_writer.add_scalar(
                    "precision", precision(epoch_outputs, epoch_labels), epoch
                )

            ### VALIDATION ###
            model.eval()
            epoch_loss = 0.0
            epoch_labels = torch.tensor([])
            epoch_outputs = torch.tensor([])
            date_from = get_date(val_loader.dataset[0][2], DATASET)
            date_to = get_date(val_loader.dataset[-1][2], DATASET)

            if epoch == 0:
                print(f"validation from {date_from} to {date_to}")

            with torch.no_grad():
                for inputs, labels, _ in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)

                    epoch_loss += outputs.shape[0] * loss.item()
                    epoch_labels = torch.cat(
                        (epoch_labels, labels.float().detach().cpu()), 0
                    )
                    epoch_outputs = torch.cat(
                        (epoch_outputs, outputs.flatten().detach().cpu()), 0
                    )

            epoch_loss = epoch_loss / len(epoch_labels)
            epoch_predictions = (epoch_outputs > 0.5).float()

            mean_f1[epoch] += f1(epoch_predictions, epoch_labels)
            mean_loss[epoch] += epoch_loss
            mean_precision[epoch] += precision(epoch_outputs, epoch_labels)
            mean_recall[epoch] += recall(epoch_outputs, epoch_labels)

            if scheduler:
                if type(scheduler) is lr_scheduler.ReduceLROnPlateau:
                    scheduler.step(loss.item())
                else:
                    scheduler.step()

    mean_loss = np.divide(mean_loss, count_folds)
    mean_f1 = np.divide(mean_f1, count_folds)
    mean_precision = np.divide(mean_precision, count_folds)
    mean_recall = np.divide(mean_recall, count_folds)

    if DEBUG:
        for idx in range(len(mean_loss)):
            validation_writer.add_scalar("loss", mean_loss[idx], idx)
            validation_writer.add_scalar("f1", mean_f1[idx], idx)
            validation_writer.add_scalar("recall", mean_recall[idx], idx)
            validation_writer.add_scalar("precision", mean_precision[idx], idx)

    return 1 - mean_f1[-1]


set_logger_config(
    level=logging.INFO,  # logging level
    log_file="./propulate.log",  # logging path
    log_to_stdout=True,  # Print log on stdout.
    log_rank=False,  # Do not prepend MPI rank to logging messages.
    colors=False,  # Use colors.
)

# CONFIG #
INFO = ""
VARIABLE = ""
NUM_EPOCHS = 0
NUM_FOLDS = 0
TENSORBOARD_PREFIX = ""
DEBUG = False
DATASET = ""

train_dataset = get_dataset(DATASET)

comm = MPI.COMM_WORLD
num_generations = 2
pop_size = 2 * MPI.COMM_WORLD.size
pop_size = MPI.COMM_WORLD.size
limits = {
    "model": ("resnet18", "resnet50", "efficientnet_s", "efficientnet_m", "inception"),
    "scheduler": ("step_01", "step_09", "plateau", "none"),
    "loss": ("bce", "bce_weighted"),
    "sampler": ("weighted_random", "none"),
    "augmentation": ("light", "heavy", "none"),
    "lr": (0.01, 0.0001),
    "batch_size": (8, 512),
    "optimizer": ("sgd_0", "sgd_09", "adam", "adagrad"),
    "dropout": (0.0, 0.9),
    "weight_decay": (0.0, 1.5),
}

rng = random.Random()

# hyperparameters from https://propulate.readthedocs.io/en/latest/tut_hpo.html
propagator = get_default_propagator(
    pop_size=pop_size,
    limits=limits,
    mate_prob=0.7,
    mut_prob=0.4,
    random_prob=0.1,
    rng=rng,
)

propulator = Propulator(
    propagator=propagator,
    loss_fn=ind_loss,
    comm=comm,
    generations=num_generations,
    rng=rng,
)

print(limits)

propulator.propulate(1, 2)
propulator.summarize(top_n=pop_size, debug=2)

print(INFO + " " + VARIABLE + " " + "done")
