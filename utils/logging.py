# %% Import

import os
import shutil
import torch
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter

# %% Methods


def setup_logging_dir(config_path, root_dir="runs", exp_name=None):

    if exp_name is None:
        exp_name = "default_experiment"

    # Create a timestamp for the experiment
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(root_dir, exp_name, timestamp)

    # Make directories if they do not exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "best_model"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "logs"), exist_ok=True)
    # os.makedirs(os.path.join(log_dir, "samples"), exist_ok=True)

    # Copy the config file to the log directory
    shutil.copy(config_path, os.path.join(log_dir, "config.yaml"))

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "logs"))

    return log_dir, writer


def save_checkpoint(state, path):
    """Save specifed 'state' package to designated path."""
    torch.save(state, path)


def config_template():
    pass
