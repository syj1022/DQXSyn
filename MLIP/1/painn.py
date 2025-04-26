import sys
from pathlib import Path
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from nff.data import Dataset, split_train_validation_test, collate_dicts, to_tensor
from nff.train import Trainer, get_model, load_model, loss, hooks, metrics, evaluate

DEVICE = 0
OUTDIR = "."
BATCH_SIZE = 32
data = os.path.basename(os.path.abspath(os.path.join(os.getcwd(), "..")))

train = torch.load(f"/scratch/alevoj1/YingjieS/DQX/data/{data}/train_dataset.pth.tar")
val = torch.load(f"/scratch/alevoj1/YingjieS/DQX/data/{data}/val_dataset.pth.tar")
test = torch.load(f"/scratch/alevoj1/YingjieS/DQX/data/{data}/test_dataset.pth.tar")

modelparams = {
    "feat_dim": 128,
    "activation": "swish",
    "n_rbf": 20,
    "cutoff": 5.0,
    "num_conv": 3,
    "output_keys": ["energy"],
    "grad_keys": ["energy_grad"],
    "skip_connection": {"energy": False},
    "learnable_k": False,
    "conv_dropout": 0.0,
    "readout_dropout": 0.0,
    "means": {"energy": train.props["energy"].mean().item()},
    "stddevs": {"energy": train.props["energy"].std().item()},
}

if os.path.isfile("best_model"):
    model = torch.load("best_model")
else:
    model = get_model(modelparams, model_type="Painn")

original_model = copy.deepcopy(model)

train_loader = DataLoader(train, batch_size=BATCH_SIZE, collate_fn=collate_dicts, sampler=RandomSampler(train))
val_loader = DataLoader(val, batch_size=BATCH_SIZE, collate_fn=collate_dicts)
test_loader = DataLoader(test, batch_size=BATCH_SIZE, collate_fn=collate_dicts)

loss_fn = loss.build_mse_loss(loss_coef={"energy_grad": 0.95, "energy": 0.05})

trainable_params = filter(lambda p: p.requires_grad, model.parameters())

optimizer = Adam(trainable_params, lr=1e-4)

train_metrics = [metrics.MeanAbsoluteError("energy"), metrics.MeanAbsoluteError("energy_grad")]

train_hooks = [
    hooks.MaxEpochHook(5000),
    hooks.CSVHook(
        OUTDIR,
        metrics=train_metrics,
    ),
    hooks.PrintingHook(OUTDIR, metrics=train_metrics, separator=" | ", time_strf="%M:%S"),
    hooks.ReduceLROnPlateauHook(
        optimizer=optimizer,
        patience=80,
        factor=0.5,
        min_lr=1e-6,
        window_length=1,
        stop_after_min=True,
    ),
]

T = Trainer(
    model_path=OUTDIR,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
    checkpoint_interval=1,
    hooks=train_hooks,
    mini_batches=1,
)


T.train(device=DEVICE, n_epochs=5000)

