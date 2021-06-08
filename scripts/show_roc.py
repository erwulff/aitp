import argparse
from pathlib import Path
import matplotlib as mpl

from learning.utils import open_config
from learning import datasets

from scripts.evaluate import compute_roc_stats, plot_roc_stats, _print_tpr_at_fpr


mpl.rc_file("my_matplotlib_rcparams")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default=None, required=True, type=str)
    parser.add_argument(
        "--evaluation_dir", "-e", default=None, required=False, type=str
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = open_config(args.config)
    if args.evaluation_dir is None:
        eval_dir = str(Path(args.checkpoint).parent.parent / "evaluation")
    else:
        eval_dir = args.evaluation_dir
    fpr, tpr, roc_auc = compute_roc_stats(eval_dir)

    dataset_class = getattr(datasets, cfg["dataset_class"])
    _print_tpr_at_fpr(fpr, tpr, labels=dataset_class.CLASS_LABELS)
    plot_roc_stats(
        fpr,
        tpr,
        roc_auc,
        save_file_path=None,
        class_labels=dataset_class.CLASS_LABELS,
        xscale="log",
    )
