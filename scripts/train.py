import os
from pathlib import Path
import argparse
import matplotlib as mpl
import shutil

import torch
from torch import nn

from torch import optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from learning.models import get_model_from_config
from learning.train_utils import fit, get_data, CheckpointSaver
from learning.utils import (
    count_trainable_parameters,
    plot_losses,
    create_train_dir,
    open_config,
    load_model,
    get_latest_checkpoint,
    delete_all_but_latest_ckpt,
)
from learning.datasets import JEDIDataset, TinyJEDIDataset, JEDIRAMDataset
from learning import datasets
from learning.transforms import SmoothLabels

from scripts.evaluate import (
    evaluate,
    compute_roc_stats,
    plot_roc_stats,
)

mpl.rc_file("my_matplotlib_rcparams")


def main(args):
    torch.set_float32_matmul_precision('high')  # enable TensorFloat32 tensor cores

    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend='nccl')  # 'nccl', 'gloo' etc.
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    torch.manual_seed(37 + seed_offset)

    cfg = open_config(args.config)
    if not ddp:
        device = cfg["device"]

    # Model
    model = get_model_from_config(cfg)  # get_model(sumO=cfg["sumO"], device=device)
    model.to(device)

    # Dataset
    data_dir = cfg["data_dir"]
    dataset_class = getattr(datasets, cfg["dataset_class"])

    if cfg["smooth_labels"]:
        transform = SmoothLabels(alpha=cfg["smooth_labels_alpha"], n_classes=len(dataset_class.CLASS_LABELS))
    else:
        transform = None

    train_dataset = dataset_class(data_dir, train=True, size=cfg["train_size"], transform=transform)
    val_dataset = dataset_class(data_dir, train=False, size=cfg["val_size"], transform=transform)

    if master_process:
        print("Trainable parameters: {}".format(count_trainable_parameters(model)))
        print("Training set size:", len(train_dataset))
        print("Validation set size:", len(val_dataset))

    # Training parameters
    dl_num_workers = cfg["dl_num_workers"]
    dl_prefectch_factor = cfg["dl_prefectch_factor"]
    bs = cfg["bs"]
    lr = cfg["lr"]
    wd = cfg["wd"]
    epochs = cfg["epochs"]
    loss_func = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1, 1, 1, 1]).to(device))

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_dl, val_dl = get_data(train_dataset, val_dataset, bs, dl_num_workers, dl_prefectch_factor, ddp)
    steps_per_epoch = ceil(len(train_dl) / ddp_world_size)  # TODO: this does not give the exact steps per epoch

    if cfg["lr_schedule"] == "constant":
        lr_scheduler = None
    elif cfg["lr_schedule"] == "onecycle":
        lr_scheduler = OneCycleLR(opt, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
    elif cfg["lr_schedule"] == "cosinedecay":
        lr_scheduler = CosineAnnealingLR(opt, T_max=steps_per_epoch * epochs)
    else:
        raise ValueError("Supported values for lr_schedule are 'constant', 'onecycle' and 'cosinedecay'.")


    if master_process:  # only do this on the head node
        train_dir = create_train_dir(args.prefix)
        shutil.copyfile(args.config, str(Path(train_dir) / Path(args.config).name))
        checkpoint_dir = train_dir / Path("checkpoints")
        checkpoint_dir.mkdir()
        checkpoint_saver = CheckpointSaver(
            save_dir=str(checkpoint_dir),
            model=model,
            optimizer=opt,
            lr_scheduler=lr_scheduler,
        )
    else:
        checkpoint_saver = None


    if cfg["compile"]:
        if master_process:
            print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    elif master_process:
        print("skipping model compilation")

    # wrap model in DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Train
    if master_process:
            print("Starting training")
    train_stats = fit(
        epochs,
        model,
        loss_func,
        opt,
        train_dl,
        val_dl,
        lr_scheduler,
        device,
        checkpoint_saver,
        ddp_rank,
    )

    # Evaluate
    if master_process:  # eval is currently only supported using 1 process
        if cfg["eval_at_train_end"]:
            evaluation_dir = train_dir / "evaluation"
            if master_process:
                evaluation_dir.mkdir()

            # Ensure we use the best performing checkpoint for evaluation
            # If we only save checkpoints when the model imrpoves we can simply get the latest checkpoint,
            # if instead we save after every epoch we need to find the best checkpoint based on val loss instead
            checkpoint_file = get_latest_checkpoint(train_dir)
            print("Checkpoint used for evaluating: {}".format(checkpoint_file))
            model = load_model(cfg, checkpoint_file)

            evaluate(model, cfg, evaluation_dir)
            plot_losses(train_stats, show=False, save_path=evaluation_dir / "loss_curves.jpg")
            fpr, tpr, roc_auc = compute_roc_stats(evaluation_dir)

            dataset_class = getattr(datasets, cfg["dataset_class"])
            plot_roc_stats(
                fpr,
                tpr,
                roc_auc,
                save_file_path=evaluation_dir / "roc.jpg",
                class_labels=dataset_class.CLASS_LABELS,
                xscale="log",
            )

        if cfg["remove_checkpoints"]:
            delete_all_but_latest_ckpt(train_dir)

    if ddp:
        destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default=None, required=True, type=str)
    parser.add_argument("--prefix", "-p", default=None, required=False, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
