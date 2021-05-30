import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path

from learning.utils import load_model, open_config
from learning.train_utils import loss_batch
from learning import datasets


def evaluate(model, config, eval_dir):
    cfg = open_config(config)
    model.eval()

    device = cfg["device"]
    dl_num_workers = cfg["dataloader_num_workers"]
    bs = cfg["bs"]

    # Dataset
    data_dir = cfg["data_dir"]
    dataset_class = getattr(datasets, cfg["dataset_class"])
    val_dataset = dataset_class(data_dir, train=False, size=cfg["val_size"])
    print("Validation set size:", len(val_dataset))

    # Training parameters
    loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 1, 1, 1]).to(device))
    val_dl = DataLoader(val_dataset, batch_size=bs * 2, num_workers=dl_num_workers, pin_memory=True)

    val_bar = tqdm(val_dl, total=len(val_dl), desc="Validation")
    with torch.no_grad():
        losses, nums = zip(*[loss_batch(model, loss_func, xb.to(device), yb.to(device)) for xb, yb in val_bar])
    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    print("Validation loss: {}".format(val_loss))
    with open(str(Path(eval_dir) / "results.txt"), "a") as f:
        f.write("Validation loss: {}\n".format(val_loss))
        f.write("model_name: {}\n".format(cfg["model_name"]))
        f.write("dataset_class: {}\n".format(cfg["dataset_class"]))
        f.write("val_size: {}\n".format(cfg["val_size"]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default=None, required=True, type=str)
    parser.add_argument("--checkpoint", "-p", default=None, required=True, type=str)
    parser.add_argument("--evaluation_dir", "-e", default=None, required=False, type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.config, args.checkpoint)
    if args.evaluation_dir is None:
        eval_dir = str(Path(args.checkpoint).parent.parent / "evaluation")
    else:
        eval_dir = args.evaluation_dir
    evaluate(model, args.config, eval_dir)