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
from data.review_processed_data import review_processed_data
from inference.evaluate import evaluate_model

# %% Methods


def parse_args():
    """Construct parser"""

    parser = argparse.ArgumentParser(description="3D-AE via PyTorch")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["prep", "review", "train", "eval"],
        required=True,
        help="Set mode to prepare data, review data, train model, or evaluate data",
    )
    parser.add_argument(
        "--data-raw",
        metavar="DIR",
        default="D:\\Large_Data\\SolarData\\171A_low_solar_activity_2012-04-10_to_2012-04-15",
        help="Path to un-processed dataset (e.g., /.tar file)",
    )
    parser.add_argument(
        "--data-clips",
        metavar="DIR",
        default="./data/processed",
        help="Path to processed data folder (e.g., /.pt files)",
    )
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        help=(
            "User requestd number of workers for training, dataloader, etc (default: 4)"
            "Actual args.num_workers dictated by script based on request for safety."
        ),
    )
    parser.add_argument(
        "--gpu-index",
        default=0,
        type=int,
        help="Use Gpu-index(0) or not if (-1)/None (default: 0).",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=torch.device,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to use for training (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--preload",
        default=False,
        type=bool,
        help="Load previously trained model from config specified checkpoint (default: False).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="None",
        type=str,
        help=(
            "Dir of desired checkpoint file within 'runs//' to load (default: 'None')."
            "Follow format: 'exp-name\\%Y-%m-%d_%H-%M-%S\\checkpoints\\checkpoint_epoch_epoch_counter.pth.tar'"
            "Or for best_model: 'exp-name\\%Y-%m-%d_%H-%M-%S\\best_model\\best_model.pth.tar'"
        ),
    )
    parser.add_argument(
        "--save-clip-stats",
        default=True,
        type=bool,
        help="Save a .csv file of processed clip stats during 'review' mode.",
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["heatmap", "boxes"],
        default="boxes",
        required=False,
        help="Set visual mode to evaluate a given video via trained model",
    )
    parser.add_argument(
        "--eval-diagnostic",
        default=False,
        type=bool,
        help="Set output of eval mode to provide diagnostic inofrmation (True) or normal visuals (deafult:False)",
    )
    parser.add_argument(
        "--model-path",
        default="./runs/anomaly-det_PaperConv3DAE_2025-07-22/2025-07-22_19-37-11",
        type=str,
        help="Path to model file following format: './runs/task_arch_YYYY-MM-DD/YYYY-MM-DD_hh-mm-ss/' ",
    )
    return parser.parse_args()


def main():
    """Main operating script"""

    args = parse_args()

    # check if gpu training is chosen
    if args.device == "cuda":
        args.num_workers = min(args.workers, os.cpu_count())
        cudnn.deterministic = True
        cudnn.benchmark = True
        print(f"[INFO] Cuda is available.")
        print("[INFO] Device Name: ", torch.cuda.get_device_name(0))
    else:
        args.num_workers = 0  # keep safe for CPU-only, avoid multiprocessing issues
        args.gpu_index = -1

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)  # load config file into local var as hierarchal dict

    # Check mode and proceed accordingly
    if args.mode == "prep":
        prepare_solar_data(args.data_raw, args.data_clips, config)
        print("[INFO] Data preparation complete.")
    elif args.mode == "review":
        review_processed_data(args.data_clips, save_stats=args.save_clip_stats)
        print("[INFO] Data review complete.")
    elif args.mode == "train":
        train_model(args, config)
        print("[INFO] Training complete & best_model selected.")
    elif args.mode == "eval":
        if args.model_path == None:
            raise ValueError(f"No Model path provided for evaluation.")
        evaluate_model(args)


# %% Script

# Only operate main() if called directly (i.e., not as module, __name__ = "my_module")
if __name__ == "__main__":

    sys.argv = [
        "main.py",
        "--mode",
        "eval",
        "--config",
        "config.yaml",
    ]  # override args for testing/debugging

    main()
