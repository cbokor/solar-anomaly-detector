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
    Initialise {train} mode for 3D-AE via PyTorch

    + inform cpu and workers
    + organise clips into a dataset
    + define a sampler
    + define train_loader (dataloader)
    + optiona(view batches)
    +
    """

    print("[INFO] Training with Device:", args.device)
    print("[INFO] Number of workers assigned:", args.num_workers)
    # print("Total number of epochs:", args.epochs)
    # print("Number of warmup epochs:", args.warmup_epochs)

    # Sampler?

    # Organise data into ImageLoader
    train_data = ClipDataSet(args.data_clips, transform=None)

    # Organise data into DataLoader:

    if args.workers > 0:
        persistent_workers = True
    else:
        persistent_workers = False

    # train_data, test_data = torch.utils.data.random_split(train_data, [**, **])
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

    # Optional batch view?:
    # utils.view_example_batch(train_loader, args.batch_size)

    # Assign model architecture from options
    model_class_name = config["model"]["model_arch"]
    model_class = getattr(conv3d_autoencoder, model_class_name)
    model = model_class()

    # Assign optimizer: default is 'Adam' with weight decay due to tiny dataset (Stable, fast, regularised for proof of concept)
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

    # Assign loss function: default is MSELoss (i.e., pixel-wise loss)
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

    pass
