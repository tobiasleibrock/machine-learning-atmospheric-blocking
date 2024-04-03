### ML ###
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from util import (
    get_dataset,
    get_model,
)

### CUSTOM MODULES ###
BATCH_SIZE = 0
WEIGHTS = ""
WEIGHTS = ""


### PREDEFINED ###
# BATCH_SIZE = 64
# WEIGHTS = "models/ukesm-geo-inception.pt"
# WEIGHTS = "models/era5-geo-inception.pt"

print(f"testing with weights: {WEIGHTS}")
device = torch.device("cuda:0")

f1 = BinaryF1Score(threshold=0.5).to(device)
recall = BinaryRecall(threshold=0.5).to(device)
precision = BinaryPrecision(threshold=0.5).to(device)

for dataset in ["era5", "ukesm", "era5-msl", "ukesm-msl"]:
    full_test_labels = torch.tensor([])
    full_test_outputs = torch.tensor([])
    tr_dataset = get_dataset(dataset)

    model = get_model("inception", 0.3, WEIGHTS)
    model.to(device)
    model.eval()

    test_loader = DataLoader(
        tr_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    with torch.no_grad():
        for inputs, labels, time in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            full_test_outputs = torch.cat(
                (full_test_outputs, outputs.flatten().detach().cpu()), 0
            )
    print(f"testing on {dataset} dataset")
    print(f"f1: {f1(full_test_outputs, torch.tensor(tr_dataset.labels))}")
    print(f"recall: {recall(full_test_outputs, torch.tensor(tr_dataset.labels))}")
    print(f"precision: {precision(full_test_outputs, torch.tensor(tr_dataset.labels))}")
