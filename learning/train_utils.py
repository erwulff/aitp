import torch
import numpy as np
from tqdm import tqdm
import time
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import get_world_size

def get_data(train_ds, val_ds, bs, num_workers=0, prefetch_factor=None, ddp=False):
    if isinstance(train_ds, torch.utils.data.Dataset):
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=True)
    else:
        train_sampler = val_sampler = None
    return (
        DataLoader(
            train_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            # pin_memory=True,
            prefetch_factor=prefetch_factor,
            sampler=train_sampler,
        ),
        DataLoader(
            val_ds,
            batch_size=bs,
            num_workers=num_workers,
            # pin_memory=True,
            prefetch_factor=prefetch_factor,
            sampler=val_sampler,
        ),
    )


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), torch.argmax(yb, dim=1))  # CrossEntropyLoss expects an index not a one-hot encoding

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(
    epochs,
    model,
    loss_func,
    opt,
    train_dl,
    val_dl,
    lr_scheduler=None,
    device="cuda",
    checkpoint_saver=None,
    ddp_rank=0,
):
    since = time.time()
    epochs_train_loss = []
    epochs_val_loss = []
    epoch_times = []
    for epoch in range(epochs):
        running_train_loss = 0.0
        epoch_start = time.perf_counter()
        model.train()
        train_bar = tqdm(train_dl, desc="Epoch {:d}/{:d}".format(epoch, epochs))
        for i, (xb, yb) in enumerate(train_bar):
            # pin x,y, allows for copying to GPU asynchronously (non_blocking=True)
            xb = xb.pin_memory().to(device, non_blocking=True)
            yb = yb.pin_memory().to(device, non_blocking=True)
            batch_loss, lenxb = loss_batch(model, loss_func, xb, yb, opt)  # ddp synchronization point (forward pass & differentiation)
            running_train_loss += np.multiply(batch_loss, lenxb)
            if i % 10 == 0:
                train_bar.set_postfix_str(s="batch_loss: {:.4f}".format(batch_loss))
                train_bar.set_description_str(desc="Epoch {:d}/{:d}".format(epoch, epochs))
            if lr_scheduler is not None:
                lr_scheduler.step()
        train_bar.close()

        model.eval()
        val_bar = tqdm(val_dl, desc="Validation")
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb.to(device), yb.to(device)) for xb, yb in val_bar])
        train_loss = running_train_loss / len(train_dl.dataset)
        epochs_train_loss.append(train_loss)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        epochs_val_loss.append(val_loss)
        current_time = time.perf_counter()
        dt = current_time - epoch_start
        epoch_times.append(dt)
        if ddp_rank == 0:
            val_bar.write(
                "Epoch: {:d} | Train Loss: {:.3f} | Val Loss: {:.3f} | Time: {}".format(
                    epoch, train_loss, val_loss, str(datetime.timedelta(seconds=round(dt)))
                )
            )
        val_bar.close()

        if checkpoint_saver is not None:
            checkpoint_saver.save(epoch, val_loss)

    time_elapsed = time.time() - since

    if ddp_rank == 0:
        print("\nTraining complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    return pd.DataFrame(
        {
            "epoch": np.arange(epochs),
            "train_loss": np.array(epochs_train_loss),
            "val_loss": np.array(epochs_val_loss),
            "epoch_times": epoch_times,
        }
    )


class CheckpointSaver:
    def __init__(self, save_dir, model, optimizer, lr_scheduler=None):
        self.save_dir = save_dir
        self.model = model
        self.opt = optimizer
        self.lr_scheduler = lr_scheduler
        self.best_val_loss = None

    def save(self, current_epoch, current_val_loss):
        if self.best_val_loss is None:
            self.best_val_loss = current_val_loss
        elif current_val_loss < self.best_val_loss:
            print("New best achieved at epoch {}: val_loss={}".format(current_epoch, current_val_loss))
            self.best_val_loss = current_val_loss
        else:
            return -1
        checkpoint_path = self.save_dir + "/checkpoint_epoch{}.pt".format(current_epoch)
        torch.save(
            {
                "epoch": current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                "loss": current_val_loss,
            },
            checkpoint_path,
        )
        print("Checkpoint saved to: {}".format(checkpoint_path))
