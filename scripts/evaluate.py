import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from learning.utils import load_model, open_config
from learning.train_utils import loss_batch
from learning import datasets


def read_eval_files(eval_dir):
    labels = np.genfromtxt(str(Path(eval_dir) / "labels.txt"), delimiter='\n')
    preds = np.genfromtxt(str(Path(eval_dir) / "predictions.txt"), delimiter='\n')
    return preds, labels

def compute_roc_stats(eval_dir):
    preds, labels = read_eval_files(eval_dir)

    labels = label_binarize(labels, classes=list(range(0, 5)))
    preds = label_binarize(preds, classes=list(range(0, 5)))

    n_classes = labels.shape[1]
    fpr = {}
    tpr = {}
    roc_auc = {}
    for ii in range(n_classes):
        fpr[ii], tpr[ii], _ = roc_curve(labels[:, ii], preds[:, ii])
        roc_auc[ii] = auc(fpr[ii], tpr[ii])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), preds.ravel())
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

def plot_roc_stats(fpr, tpr, roc_auc, save_file_path=None, legend_fontsize="x-large"):
    mpl.rc_file("my_matplotlib_rcparams")
    n_classes = len(fpr.keys()) - 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    for ii in range(n_classes):
        plt.plot(fpr[ii], tpr[ii],
                 label='Class {0} (area = {1:0.2f})'.format(ii, roc_auc[ii]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(fontsize=legend_fontsize)
    if save_file_path is not None:
        plt.savefig(save_file_path)
    else:
        plt.show()


def evaluate(model, config, eval_dir):

    Path(eval_dir).mkdir(parents=True, exist_ok=True)

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
        losses = []
        nums = []
        predictions_file = open(str(Path(eval_dir) / "predictions.txt"), 'w', newline='')
        # predictions_writer = csv.writer(predictions_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        label_file = open(str(Path(eval_dir) / "labels.txt"), 'w', newline='')
        # label_writer = csv.writer(label_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for xb, yb in val_bar:
            confidenceb = model(xb.to(device))
            predb = torch.argmax(confidenceb, dim=1)
            labelb = torch.argmax(yb.to(device), dim=1)  # CrossEntropyLoss expects an index not a one-hot encoding
            loss = loss_func(confidenceb, labelb).item()

            losses.append(loss)
            nums.append(len(xb))

            # predictions_writer.writerows(predb.numpy())
            # label_writer.writerows(labelb.numpy())
            sep = "\n"
            predictions_file.write(np.array2string(predb.numpy(), separator=sep)[1:-1].replace(" ", "") + sep)
            label_file.write(np.array2string(labelb.numpy(), separator=sep)[1:-1].replace(" ", "") + sep)

        predictions_file.close()
        label_file.close()

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
    fpr, tpr, roc_auc = compute_roc_stats(eval_dir)
    plot_roc_stats(fpr, tpr, roc_auc, save_file_path=Path(eval_dir) / "roc.jpg")