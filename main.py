# %% Tensorboard setep
# (1) tensorboard --logdir="...<ThisRepoFolder>\runs"
# (2) http://localhost:6006

# %% Imports

import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
import sys

from data.prepare_data import prepare_solar_data
from training.train import train_model

# from inference.evaluate import evaluate_model

# %% Initialize


# %% Methods


def parse_args():
    """Construct parser"""
    parser = argparse.ArgumentParser(description="3D-AE via PyTorch")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prep", "train", "eval"],
        required=True,
        help="set mode to evaluate or train model",
    )
    parser.add_argument(
        "-data-raw",
        metavar="DIR",
        default="D:\\Large_Data\\SolarData\\aia_lev1_4k_304A_100recordingsFrom1stNov",
        help="path to un-processed dataset (e.g., /.tar file)",
    )
    parser.add_argument(
        "-data-clips",
        metavar="DIR",
        default="./data/processed",
        help="path to processed data folder (e.g., /.pt files)",
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        help="number of workers for dataloader (default: 4)",
    )
    parser.add_argument(
        "--gpu-index",
        default=0,
        type=int,
        help="Use Gpu-index(0) or not if (-1)/None (default: 0).",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to use for training (default: cuda if available, else cpu)",
    )
    # do these two make sense without a backbone? maintain for now
    parser.add_argument(
        "--pretrain",
        default=True,
        type=bool,
        help="pretrain model from own data before loading from checkpoint",
    )
    parser.add_argument(
        "--preload",
        default=False,
        type=bool,
        help="preload previously trained model (default: False)",
    )
    return parser.parse_args()


def main():
    """Main operating script"""

    args = parse_args()

    # check if gpu training is available
    if torch.cuda.is_available():
        args.num_workers = min(args.workers, os.cpu_count())
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.num_workers = 0  # keep safe for CPU-only, avoid multiprocessing issues
        args.gpu_index = -1

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)  # load config file into local var

    # Check mode and proceed accordingly
    if args.mode == "prep":
        prepare_solar_data(args.data_raw, args.data, config)
        print("[INFO] Data preparation complete.")
    elif args.mode == "train":
        train_model(args, config)
    # elif args.mode == "eval":
    #    evaluate_model(config)

    pass


# %% Script

# Only operate main() if called directly (i.e., not as module, __name__ = "my_module")
if __name__ == "__main__":

    sys.argv = [
        "main.py",
        "--mode",
        "train",
        "--config",
        "config.yaml",
    ]  # override args for testing/debugging

    main()
