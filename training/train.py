# %% Goal
"""
### Baseline Architecture Notes (DEL later): 3D Conv Autoencoder

+ Assume individual clips currently stored as (C, T, H, W) tensors
+ next need to convert to (B, C, T, H, W) tensors in a dataloader

+ Input: (B, C, T, H, W)
-> Conv3D layers
-> ReLU + MaxPool3D
-> Latent Space
-> ConvTranspose3D layers
-> ReLU + Sigmoid
+ Output: Reconstructed clip (B, C, T, H, W)

- **Loss Function**: MSE (or SSIM later)

"""

# %% Import

import os
import torch
import models.conv3d_autoencoder as conv3d_autoencoder

from torch.utils.data import DataLoader
from data.data_aug import ClipDataSet
from training.anomaly_detection import AnomalyDetection
from training.loss_functions import LOSS_REGISTRY

# %% Methods


def train_model(args, config):
    """
    Initialize and train a 3D Autoencoder for video anomaly detection using PyTorch.

    This function sets up the full training pipeline including:
    - Device and worker configuration
    - Dataset and DataLoader initialization
    - Model instantiation from config
    - Optimizer and scheduler setup
    - Loss function selection from registry
    - Optional checkpoint loading
    - Execution of the training loop

    Args:
        args (Namespace): Command-line arguments or config object with attributes such as:
            - device (str): Target device ('cuda' or 'cpu')
            - num_workers (int): Number of workers for training, dataloader, etc.
            - data_clips (str or Path): Path to training clip directory
            - gpu_index (int): GPU index to use (set -1 or None to disable CUDA context setting)
            - checkpoint_dir (str): Directory to load checkpoint from
            - preload (bool): Whether to preload model from checkpoint
        config (dict): Nested configuration dictionary containing:
            - model: Model architecture and loss function settings
            - optimizer: Optimizer hyperparameters
            - training: Scheduler and training loop settings
            - data_loader: Batch size and DataLoader flags

    Notes:
        - The model class must be defined in `conv3d_autoencoder` and match the name in `config["model"]["model_arch"]`.
        - The loss function must be registered in `LOSS_REGISTRY`.
        - Training is handled via the `AnomalyDetection.train_reconstrustive()` method.

    Returns:
        None
    """

    print("[INFO] Training with Device:", args.device)
    print("[INFO] Number of workers assigned:", args.num_workers)

    # Sampler - placeholder

    # Organise data into ImageLoader
    train_data = ClipDataSet(args.data_clips, transform=None)

    # Using persistent_workers=True (if more than 1 worker) avoids deleting and recreating the workers accross epochs
    if args.num_workers > 0:
        persistent_workers = True
    else:
        persistent_workers = False

    # Organise data into DataLoader
    train_loader = DataLoader(
        train_data,
        batch_size=int(config["data_loader"]["batch_size"]),
        drop_last=bool(config["data_loader"]["drop_last"]),
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=persistent_workers,
        # sampler=sampler,
    )

    # Assign model architecture from options
    model_class_name = config["model"]["model_arch"]
    model_class = getattr(conv3d_autoencoder, model_class_name)
    model = model_class()

    # Assign optimizer: default is 'AdamW' for stabuility
    optimizer = torch.optim.AdamW(
        model.parameters(),
        float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )

    # Scheduler to dictate the learning rate with CosineAnnealingLR (i.e., learning rate annealed accros args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["epochs"]),
        eta_min=float(config["training"]["eta_min"]),
        last_epoch=-1,
    )

    # Assign loss function based on config choice
    criterion_class_name = config["model"]["loss_function"]
    criterion_class = LOSS_REGISTRY[criterion_class_name]
    if "loss_params" in config["model"]:
        criterion = criterion_class(
            **config["model"]["loss_params"]
        )  # remeber, ** is dic unpacker
    else:
        criterion = criterion_class()

    # Context manager to temporarily set active CUDA device. no-op if gpu_index = negative(int) or None.
    with torch.cuda.device(args.gpu_index):

        # Assemble full anomaly detection pipeline
        anomaly_ae = AnomalyDetection(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            config=config,
            device=args.device,
        )

        # Load prior checkpoint if exists & requested
        if (
            os.path.exists(os.path.join("runs", args.checkpoint_dir))
            and args.preload == True
        ):
            anomaly_ae.extract_checkpoint(args.checkpoint_dir)

        # Train full pipeline
        anomaly_ae.train_reconstrustive(config, train_loader)
