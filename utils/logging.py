# %% Import

import numpy as np

# --- Monkey-patch for NumPy >= 1.24 compatibility ---
if not hasattr(np, "bool8"):
    np.bool8 = (
        np.bool_
    )  # TODO: Remove once summarywritter/tensorboard dependencies stop using np.bool8
# ==================================

import os
import shutil
import torch
import webbrowser
from tensorboard import program
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter

# %% Methods


def launch_tensorboard(log_dir="runs", port=6006, open_browser=True):
    """
    Launch TensorBoard with monkey-patched numpy.bool8 for compatibility.

    Parameters:
    - log_dir (str): Path to the TensorBoard log directory.
    - port (int): Port to serve TensorBoard on.
    - open_browser (bool): Whether to open the TensorBoard URL in a browser.
    """
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", log_dir, "--port", str(port)])
    url = tb.launch

    print(f"[TensorBoard] Running at: {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"[Tensorboard] Could not open browser: {e}")


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
