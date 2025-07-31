# %% Import

# temporary patch to solve enviroment conflict. summarywritter needs bool8, which was removed after numpy 1.2.3
# need to downgrade numpy to 1.2.3.5 once download script done
# ==============================
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
# ==============================

import os
import shutil
import torch
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter

# %% Methods


def setup_logging_dir(config_path, root_dir="runs", exp_name=None):
    """
    Set up a directory structure for logging and checkpointing during training.

    This function creates a timestamped experiment directory under the specified root,
    copies the given configuration file to that directory, and initializes a TensorBoard
    `SummaryWriter` for logging.

    The following subdirectories are created:
        - checkpoints: for storing periodic model checkpoints
        - best_model: for storing the best performing model
        - logs: for TensorBoard logs

    Parameters
    ----------
    config_path : str
        Path to the configuration YAML file to be copied into the log directory.
    root_dir : str, optional
        Root directory under which experiment folders will be created (default is "runs").
    exp_name : str, optional
        Custom experiment name. If None, defaults to "default_experiment".

    Returns
    -------
    log_dir : str
        Full path to the created experiment logging directory.
    writer : SummaryWriter
        TensorBoard SummaryWriter initialized for the experiment.

    Side Effects
    ------------
    - Creates directories for logs and checkpoints if they don't exist.
    - Copies the config file to the log directory.
    """

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
