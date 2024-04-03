### ML ###
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from torchvision.models.inception import Inception3
from util import (
    get_dataset,
    get_optimizer,
    get_scheduler,
    get_model,
    get_transform,
    get_date,
)

### GLOBAL ###
INFO = ""
SINFO = ""

### CONFIGURATION ###
BATCH_SIZE = 0
LEARNING_RATE = 0
DROPOUT = 0
MOMENTUM = 0
WEIGHT_DECAY = 0
EPOCHS = 0
TRANSFORM = ""
LOSS = ""
OPTIMIZER = ""
SCHEDULER = ""
MODEL = ""
DEBUG = False
TRAIN_DATASET = ""
TEST_DATASET = ""

#### Train: UKESM Test: ERA5 #####
# BATCH_SIZE = 121
# LEARNING_RATE = 0.00667
# DROPOUT = 0.118
# MOMENTUM = 0.0
# WEIGHT_DECAY = 0.126
# EPOCHS = 30
# TRANSFORM = "light", "none"
# LOSS = "bce_weighted"
# OPTIMIZER = "adagrad"
# SCHEDULER = "step_01"
# MODEL = "inception"
# DEBUG = True
# TRAIN_DATASET = "ukesm"
# TEST_DATASET = "era5"

#### Train: ERA5 Test: UKESM #####
# BATCH_SIZE = 157
# LEARNING_RATE = 0.00419
# DROPOUT = 0.00506
# MOMENTUM = 0.9
# WEIGHT_DECAY = 0.47
# EPOCHS = 30
# TRANSFORM = "heavy"
# LOSS = "bce_weighted"
# OPTIMIZER = "adagrad"
# SCHEDULER = "step_01"
# MODEL = "inception"
# DEBUG = False
# TRAIN_DATASET = "era5"
# TEST_DATASET = "ukesm"


def train_model(model, optimizer, scheduler, train_dataset, test_dataset, num_epochs):
    if DEBUG:
        year = get_date(test_dataset[0][2], TEST_DATASET).year
        test_writer = SummaryWriter(f"{INFO}/te/{str(year)}/")
        training_writer = SummaryWriter(f"{INFO}/tr/{str(year)}/")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(num_epochs):
        print(f"epoch {epoch+1}/{num_epochs}")
        ### TRAINING ###
        model.train()
        epoch_loss = 0.0
        epoch_labels = torch.tensor([])
        epoch_outputs = torch.tensor([])
        for inputs, labels, time in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # fix for inception model in pytorch
            # https://discuss.pytorch.org/t/inception-v3-is-not-working-very-well/38296/3
            if type(model) is Inception3:
                outputs = model(inputs.float())[0]
            else:
                outputs = model(inputs.float())

            if LOSS == "bce_weighted":
                class_counts = torch.bincount(labels.long())
                class_weights = BATCH_SIZE / (2.0 * class_counts.float())
                sample_weights = class_weights[labels.long()]
                criterion = nn.BCELoss(weight=sample_weights)
            else:
                criterion = nn.BCELoss()

            loss = criterion(outputs.flatten(), labels.float())
            loss.backward()
            optimizer.step()

            epoch_loss += outputs.shape[0] * loss.item()
            epoch_labels = torch.cat((epoch_labels, labels.float().detach().cpu()), 0)
            epoch_outputs = torch.cat(
                (epoch_outputs, outputs.flatten().detach().cpu()), 0
            )

        epoch_loss = epoch_loss / len(epoch_labels)
        print(f"trn f1 {f1(epoch_outputs, epoch_labels)}")

        if DEBUG:
            training_writer.add_scalar("loss", epoch_loss, epoch)
            training_writer.add_scalar(
                "recall", recall(epoch_outputs, epoch_labels), epoch
            )
            training_writer.add_scalar(
                "precision", precision(epoch_outputs, epoch_labels), epoch
            )
            training_writer.add_scalar("f1", f1(epoch_outputs, epoch_labels), epoch)

        scheduler.step()

    ### TEST ###
    model.eval()
    epoch_loss = 0.0
    epoch_labels = torch.tensor([])
    epoch_outputs = torch.tensor([])
    with torch.no_grad():
        for inputs, labels, time in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            epoch_loss += outputs.shape[0] * loss.item()
            epoch_labels = torch.cat((epoch_labels, labels.float().detach().cpu()), 0)
            epoch_outputs = torch.cat(
                (epoch_outputs, outputs.flatten().detach().cpu()), 0
            )

    epoch_loss = epoch_loss / len(epoch_labels)
    print(f"tst f1 {f1(epoch_outputs, epoch_labels)}")
    date_from = get_date(test_dataset[0][2], TEST_DATASET)
    date_to = get_date(test_dataset[-1][2], TEST_DATASET)

    print(f"test from {date_from} to {date_to}")
    print("-----------------------------------")

    if DEBUG:
        test_writer.add_scalar("loss", epoch_loss, epoch)
        test_writer.add_scalar("recall", recall(epoch_outputs, epoch_labels), epoch)
        test_writer.add_scalar(
            "precision", precision(epoch_outputs, epoch_labels), epoch
        )
        test_writer.add_scalar("f1", f1(epoch_outputs, epoch_labels), epoch)

    # CONFUSION MATRIX
    if DEBUG:
        conf_matrix = confusion_matrix(
            epoch_labels,
            (epoch_outputs >= 0.5).int(),
            labels=np.array([0, 1]),
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix,
            display_labels=["no blocking", "blocking"],
        )
        disp.plot()
        test_writer.add_figure("conf-matrix", disp.figure_, global_step=epoch)

    print("TEST METRICS")
    print(INFO)
    print(f"f1: {f1(epoch_outputs, epoch_labels)}")
    print(f"recall: {recall(epoch_outputs, epoch_labels)}")
    print(f"precision: {precision(epoch_outputs, epoch_labels)}")

    return model


device = torch.device("cuda:0")

f1 = BinaryF1Score(threshold=0.5).to(device)
recall = BinaryRecall(threshold=0.5).to(device)
precision = BinaryPrecision(threshold=0.5).to(device)

tr_dataset = get_dataset(TRAIN_DATASET)
ts_dataset = get_dataset(TEST_DATASET)

model = get_model(MODEL, DROPOUT)
model.to(device)

transform = get_transform(TRANSFORM)
optimizer = get_optimizer(OPTIMIZER, WEIGHT_DECAY, LEARNING_RATE, model)
scheduler = get_scheduler(SCHEDULER, optimizer)

model = train_model(
    model,
    optimizer,
    scheduler,
    tr_dataset,
    ts_dataset,
    EPOCHS,
)

torch.save(model.state_dict(), f"./models/{SINFO}.pt")
