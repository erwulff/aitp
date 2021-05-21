import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import torch


def plot_losses(df, show=False, save_path=None, skip=0):
    plt.figure()
    plt.plot(df["epoch"][skip:], df["train_loss"][skip:], label="train loss", marker=".", linestyle="-")
    plt.plot(df["epoch"][skip:], df["val_loss"][skip:], label="valid loss", marker=".", linestyle="-")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_train_dir():
    train_dir = Path("training_sessions") / Path(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    train_dir.mkdir(parents=True)
    return train_dir


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
