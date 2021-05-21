from pathlib import Path
import argparse
import yaml
import matplotlib as mpl

import torch
from torch import nn

from torch import optim
from torch.optim.lr_scheduler import OneCycleLR

from learning.models import GraphNet, get_model
from learning.train_utils import fit, get_data, CheckpointSaver
from learning.utils import count_trainable_parameters, plot_losses, create_train_dir
from learning.datasets import JEDIDataset, TinyJEDIDataset, JEDIRAMDataset

mpl.rc_file("scripts/my_matplotlib_rcparams")


def main(args):

    with open(args.config, "r") as ymlfile:
        cfg = yaml.load(ymlfile)

    # Model
    jedinet = get_model()
    print("Trainable parameters: {}".format(count_trainable_parameters(jedinet)))

    # Dataset
    data_dir = "../data/hls4ml_LHCjet_150p/hls4ml_LHCjet_150p_JEDI"
    # train_dataset = TinyJEDIDataset(data_dir, train=True, size=200)
    # val_dataset = TinyJEDIDataset(data_dir, train=False, size=200)
    train_dataset = JEDIDataset(data_dir, train=True)
    val_dataset = JEDIDataset(data_dir, train=False)
    print("Training set size:", len(train_dataset))
    print("Validation set size:", len(val_dataset))

    # Training parameters
    device = cfg["device"]
    dl_num_workers = cfg["dataloader_num_workers"]
    bs = cfg["bs"]
    lr = cfg["lr"]
    wd = cfg["wd"]
    epochs = cfg["epochs"]
    loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 1, 1, 1]))

    opt = optim.AdamW(jedinet.parameters(), lr=lr, weight_decay=wd)
    train_dl, val_dl = get_data(train_dataset, val_dataset, bs, dl_num_workers)
    onecycle_scheduler = OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(train_dl), epochs=epochs)

    train_dir = create_train_dir()
    checkpoint_dir = train_dir / Path("checkpoints")
    checkpoint_dir.mkdir()
    checkpoint_saver = CheckpointSaver(
        save_dir=str(checkpoint_dir),
        model=jedinet,
        optimizer=opt,
        lr_scheduler=onecycle_scheduler,
    )

    # Train
    train_stats = fit(epochs, jedinet, loss_func, opt, train_dl, val_dl, onecycle_scheduler, device, checkpoint_saver)

    # Analysis
    print(train_stats)
    evaluation_dir = train_dir / "evaluation"
    evaluation_dir.mkdir()
    plot_losses(train_stats, show=False, save_path=evaluation_dir / "loss_curves.jpg")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default=None, required=True, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
