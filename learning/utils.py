import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import torch
from learning.models import get_model_from_config
import yaml

def open_config(config_file):
    with open(config_file, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg

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


def create_train_dir(prefix):
    if prefix is None:
        train_dir = Path("training_sessions") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        train_dir = Path("training_sessions") / (prefix + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    train_dir.mkdir(parents=True)
    return train_dir


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def load_model(cfg, checkpoint, device=None):
    model = get_model_from_config(cfg)
    checkpoint = torch.load(checkpoint, map_location=device or torch.device(cfg["device"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
