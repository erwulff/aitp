import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv

from pandas import DataFrame

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from learning.utils import load_model, open_config, get_latest_checkpoint, delete_all_but_latest_ckpt
from learning.train_utils import loss_batch
from learning import datasets


mpl.rc_file("my_matplotlib_rcparams")


def read_eval_files(eval_dir):
    labels = np.genfromtxt(str(Path(eval_dir) / "labels.csv"), delimiter=" ")
    logits = np.genfromtxt(str(Path(eval_dir) / "predictions.csv"), delimiter=" ")
    return logits, labels


def compute_roc_stats(eval_dir):
    logits, labels = read_eval_files(eval_dir)

    n_classes = labels.shape[1]
    fpr = {}
    tpr = {}
    roc_auc = {}
    for ii in range(n_classes):
        fpr[ii], tpr[ii], _ = roc_curve(labels[:, ii], logits[:, ii])
        roc_auc[ii] = auc(fpr[ii], tpr[ii])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def plot_roc_stats(
    fpr,
    tpr,
    roc_auc,
    save_file_path=None,
    legend_fontsize="x-large",
    xscale="linear",
    class_labels=None,
    show_averages=False,
):
    n_classes = len(fpr.keys()) - 2
    plt.figure()

    if show_averages:
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average (area = {0:0.4f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average (area = {0:0.4f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

    for ii in range(n_classes):
        label = "Class {0}" if class_labels is None else class_labels[ii]
        label = label + ": AUC = {:0.4f}".format(roc_auc[ii])
        plt.plot(fpr[ii], tpr[ii], label=label)

    if xscale == "linear":
        plt.xlim([0.0, 1.0])
        plt.plot([0, 1], [0, 1], "k--")
    elif xscale == "log":
        plt.xlim([5e-4, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xscale(xscale)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(fontsize=legend_fontsize, loc="best")
    if save_file_path is not None:
        plt.savefig(save_file_path)
    else:
        plt.show()


def _print_tpr_at_fpr(fpr, tpr, labels, suppress=False):
    tpr_at_fpr10 = {}
    tpr_at_fpr1 = {}
    for ii in range(len(labels)):
        tpr_at_fpr10[ii] = np.interp(0.1, fpr[ii], tpr[ii])
        tpr_at_fpr1[ii] = np.interp(0.01, fpr[ii], tpr[ii])
    tpr_fpr_summary = DataFrame([tpr_at_fpr10, tpr_at_fpr1], index=["FPR=10%", "FPR= 1%"])
    tpr_fpr_summary.rename(
        columns={x: key for x, key in zip(range(len(labels)), labels)},
        inplace=True,
    )
    paper_results = DataFrame(
        {
            "gluon": [0.879, 0.482],
            "light quark": [0.818, 0.301],
            "W": [0.927, 0.658],
            "Z": [0.903, 0.729],
            "top": [0.931, 0.632],
        },
        index=["FPR=10%", "FPR= 1%"],
    )
    diff = tpr_fpr_summary - paper_results

    if not suppress:
        print(tpr_fpr_summary)
        print("Difference compared to JEDI-net paper:")
        print(diff)

    return tpr_at_fpr10, tpr_at_fpr1, tpr_fpr_summary, diff


def evaluate(model, cfg, eval_dir):

    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

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

    val_bar = tqdm(val_dl, total=len(val_dl), desc="Evaluating")
    with torch.no_grad():
        losses = []
        nums = []
        predictions_file = open(str(eval_dir / "predictions.csv"), "w", newline="")
        predictions_writer = csv.writer(predictions_file, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)

        label_file = open(str(eval_dir / "labels.csv"), "w", newline="")
        label_writer = csv.writer(label_file, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)

        for xb, yb in val_bar:
            logitsb = model(xb.to(device))
            yb = yb.to(device)
            labelb = torch.argmax(yb, dim=1)  # CrossEntropyLoss expects an index not a one-hot encoding
            loss = loss_func(logitsb, labelb).item()

            losses.append(loss)
            nums.append(len(xb))

            for logits, y in zip(logitsb.cpu().numpy(), yb.cpu().numpy()):
                predictions_writer.writerow(logits)
                label_writer.writerow(y)

        predictions_file.close()
        label_file.close()

    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

    fpr, tpr, roc_auc = compute_roc_stats(eval_dir)

    dataset_class = getattr(datasets, cfg["dataset_class"])
    _, _, tpr_fpr_summary, diff = _print_tpr_at_fpr(fpr, tpr, labels=dataset_class.CLASS_LABELS)
    plot_roc_stats(
        fpr,
        tpr,
        roc_auc,
        save_file_path=eval_dir / "roc.jpg",
        class_labels=dataset_class.CLASS_LABELS,
        xscale="log",
    )

    print("Validation loss: {}".format(val_loss))
    with open(str(eval_dir / "results.txt"), "a") as f:
        f.write("Validation loss: {}\n".format(val_loss))
        f.write("model_name: {}\n".format(cfg["model_name"]))
        f.write("dataset_class: {}\n".format(cfg["dataset_class"]))
        f.write("val_size: {}\n".format(cfg["val_size"]))
        f.write("\n" + tpr_fpr_summary.to_string() + "\n")
        f.write("Difference compared to JEDI-net paper:\n")
        f.write(diff.to_string() + "\n")

    print("Evaluation done.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default=None, required=True, type=str)
    parser.add_argument("--checkpoint", "-p", default=None, required=False, type=str)
    parser.add_argument("--evaluation_dir", "-e", default=None, required=False, type=str)
    parser.add_argument("--train_dir", "-t", default=None, required=True, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = open_config(args.config)
    train_dir = Path(args.train_dir)
    if args.evaluation_dir is None:
        eval_dir = str(train_dir / "evaluation")
    else:
        eval_dir = args.evaluation_dir

    if args.checkpoint is not None:
        checkpoint_file = args.checkpoint
    else:
        # If we only save checkpoints when the model imrpoves we can simply get the latest checkpoint,
        # if instead we save after every epoch we need to find the best checkpoint based on val loss instead
        checkpoint_file = get_latest_checkpoint(train_dir)

    print("Loading checkpoint file from: {}".format(checkpoint_file))
    model = load_model(cfg, checkpoint_file)
    evaluate(model, cfg, eval_dir)
    if cfg["remove_checkpoints"]:
        delete_all_but_latest_ckpt(train_dir)
