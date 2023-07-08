import argparse
import os
import pathlib
from typing import Dict, Optional, Tuple
import numpy as np

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from tqdm.auto import tqdm
from pathlib import Path

import sys
import inspect

from data.knee_data import KneeDataClassificationModule
from rss_module import RSS
import fire
import itertools
import random


def get_model(
    args: argparse.ArgumentParser, device: torch.device,
) -> pl.LightningModule:
    if args.data_type == "knee":
        model = RSS(
            args,
            image_shape=[320, 320],
            kspace_shape=[640, 400],
            device=device,
        )
    else:
        raise NotImplementedError
    return model

def train_model(
    args: argparse.Namespace,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    device: torch.device,
) -> pl.LightningModule:

    log_dir = (
        Path(args.log_dir)
        / args.data_type
        / str(args.n_seed)
        / str(args.lr)
        / str(args.weight_decay)
    )
    noise = [0]
    model_dir =  str(args.model_dir) + '/'  + str(args.n_seed) + '/' + str(args.lr) + '/' + str(args.weight_decay)

    if not os.path.isdir(str(log_dir)):
        try :
            os.makedirs(str(log_dir))
        except :
            print(f"Directory {str(log_dir)} already exists")
    if not os.path.isdir(str(model_dir)):
        try :
            os.makedirs(str(model_dir))
        except :
            print(f"Directory {str(model_dir)} already exists")

    csv_logger = CSVLogger(save_dir=log_dir, name=f"train_noise-{args.noise_percent}", version=f"{args.n_seed}")
    wandb_logger = WandbLogger(name=f"{args.lr}-{args.weight_decay}", project=args.wandb_project, offline=args.wandb_offline)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model_checkpoint = ModelCheckpoint(monitor='val_auc_mean', dirpath=model_dir, filename="{epoch:02d}-{val_auc_mean:.2f}",save_top_k=1, mode='max')
    early_stop_callback = EarlyStopping(monitor='val_auc_mean', patience=5, mode='max', log_rank_zero_only=True)

    trainer: pl.Trainer = pl.Trainer(
        accelerator="auto", 
        devices="auto",
        max_epochs=args.n_epochs,
        logger=[wandb_logger, csv_logger],
        callbacks=[model_checkpoint, early_stop_callback, lr_monitor],
        sync_batchnorm=args.sync_batchnorm
    )
    if args.skip_training :
        datamodule.setup()
        trainer.validate(model,datamodule.val_dataloader())
    else :
        trainer.fit(model, datamodule)

    return model


def test_model(
    args: argparse.Namespace,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    device: torch.device,
) -> pl.LightningModule:
    
    model_dir = str(args.model_dir) + '/'  + str(args.n_seed) + '/' + str(args.lr) + '/' + str(args.weight_decay)
    checkpoint_filename = os.listdir(model_dir)[0]
    print("Checkpoint file: ", model_dir, checkpoint_filename)
    log_dir = (
        Path(args.log_dir)
        / args.data_type
    )
    csv_logger = CSVLogger(save_dir=log_dir, name=f"test_noise-{args.noise_type}-{args.noise_percent}", version=f"{args.n_seed}")
    model = RSS.load_from_checkpoint(model_dir + '/' + checkpoint_filename, norm=args.norm)
    trainer = pl.Trainer(gpus=1 if str(device).startswith("cuda") else 0, logger=csv_logger)

    model.eval()
    M_val = trainer.validate(model, datamodule.val_dataloader())  
    M = trainer.test(model, datamodule.test_dataloader())


def get_args():
    parser = argparse.ArgumentParser(description="Indirect MR Screener training")
    # logging parameters
    # parser.add_argument("--model_dir", type=str)
    # parser.add_argument("--log_dir", type=str)
    parser.add_argument("--model_dir", type=str, default="/vast/dm5182/scratch_files/midl_runs/paper_runs/trained_models_ln/knee")
    parser.add_argument("--log_dir", type=str, default="/vast/dm5182/scratch_files/midl_runs/reb_logs/trained_logs_ln_comb/knee")

    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--dev_mode", action="store_true")
    parser.add_argument("--wandb_project", type=str)
    
    # data parameters
    parser.add_argument(
        "--data_type", type=str, default="knee",
    )
    parser.add_argument(
        "--task", type=str, default="classification",
    )
    parser.add_argument("--image_shape", type=int, default=[320, 320], nargs=2, required=False)
    parser.add_argument("--image_type", type=str, default='orig', required=False, choices=["orig"])
    parser.add_argument("--split_csv_file", type=str, default='/vast/dm5182/datasets/fastmri/single_coil/processed_data/knee/splits/metadata_knee_sc.csv')
    parser.add_argument("--sampler_filename", type=str, default="/vast/dm5182/datasets/fastmri/single_coil/processed_data/knee/sampler_knee_tr.p")
    parser.add_argument(
        "--model_type",
        type=str,
        default="preact_resnet18",
        choices=[
            "preact_resnet18",
            "preact_resnet34",
            "preact_resnet50",
            "preact_resnet101",
            "preact_resnet152",
        ],
    )

    # training parameters
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--n_seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--drop_prob", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--noise_percent", type=int, default=0)

    parser.add_argument("--n_masks", type=int, default=100)

    parser.add_argument("--sweep_step", type=int)
    parser.add_argument('--debug',  action='store_true')
    parser.add_argument('--downsample', action='store_true')
    parser.add_argument('--wandb_offline', action='store_true')
    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--sync_batchnorm', action="store_true")
    parser.add_argument("--norm", type=str, default="group", required=False, choices=["group", "batch", "layer"],)
    parser.add_argument("--noise_type", type=str, default="rice", choices=["rice", "ghosting", "spike", "field", "anisotropy", "elastic", "motion"])

    args, unkown = parser.parse_known_args()
    
    return args


def retreve_config(args, sweep_step=None):
    if sweep_step is None :
        return args
    grid = {
        "noise_percent": [0],
        "n_seed" : [1,2,3,4,5], 
        "lr" : [1e-3,1e-4,1e-5,1e-6],
        "weight_decay" : [0.1,0.01,0.001,0.0001,0.00001],
    }

    grid_setups = list(
        dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())
    )
    step_grid = grid_setups[sweep_step - 1]  
    if torch.cuda.device_count() > 0:
        expr_device = "cuda"
    else:
        expr_device = "cpu"
    
    args.lr = step_grid["lr"]
    args.weight_decay = step_grid["weight_decay"]
    args.margin = step_grid["margin"]
    args.sweep_step = sweep_step
    args.noise_percent = step_grid["noise_percent"]
    args.n_seed = step_grid['n_seed']

    return args


def run_experiment(args):
    
    print(args, flush=True)
    if torch.cuda.is_available():
        print("Found CUDA device, running job on GPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datamodule =  KneeDataClassificationModule(args, label_type="knee")
    model = get_model(args, device)
    if args.mode == "train":
        model = train_model(args=args, model=model, datamodule=datamodule, device=device,)
    else:
        datamodule.setup()
        test_model(args=args, model=model, datamodule=datamodule, device=device,)

def main(sweep_step=None):
    args = get_args()
    config = retreve_config(args, sweep_step)
    run_experiment(config)


if __name__ == "__main__":
     fire.Fire(main)
