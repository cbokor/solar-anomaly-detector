# %% Imports

import torch
import utils.logging as logging
import os
from tqdm import tqdm
from datetime import datetime
from torch.amp.grad_scaler import GradScaler

# %% Jobs
"""
priority:

low_priority:
- tensorboard and protobuff dependancy tug of war!??!?!?! COMPLETE but! maybe worth reading into
- apparently conda channels are often months behind the latest PyPI packages, research 'pip' and 'PyPI'
- my graphics card is low on vram for CNN work, currently using cpu. for gpu try batch_size=2 only with AMP 
    and gradient accumulation (otherwise batch_size=1, batch_size>2 is likely to generate OOP errors)
- currently doing writter logs per epoch, checkpoints based on division of epoch (may want per iter in future)
- form a config class to formalise the config.yaml file format, enabling automated experiemnt tracking (do at end of build)
- if i update and checkpoint logging, will need to update extract_checkpoint
"""

# %% Class


class AnomalyDetection:
    """
    Wrapper class for managing anomaly detection training pipeline components including model, optimizer,
    scheduler, loss function, and writter logging.
    """

    def __init__(self, model, optimizer, scheduler, criterion, config, device):
        """Instance input properties and initiate SummaryWriter"""
        self.model = model.to(device, non_blocking=True)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.device = device

        timestamp = datetime.now().strftime("%Y-%m-%d")
        model_arch = config["model"]["model_arch"]
        self.log_dir, self.writer = logging.setup_logging_dir(
            config_path="config.yaml",
            root_dir="runs",
            exp_name=f"anomaly-det_{model_arch}_{timestamp}",
        )

    def extract_checkpoint(self, checkpoint_dir):
        """load previous specified sim_clr checkpoint into encoder, default: args.checkpoint_path"""
        checkpoint_path = os.path.join("runs", checkpoint_dir)
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"] + 1
        self.config["training"]["start_epoch"] = self.epoch

        # Optional: Load scheduler & best loss
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "best_train_loss" in checkpoint:
            self.config.best_loss = checkpoint["best_loss"]

        print(f"Resumed from epoch {self.epoch}")

    def train_reconstrustive(self, config, train_loader):
        """
        main training loop for chosen reconstructive architecture
        optional AutoMixedPrecision
        """

        # Class to handle dynamic loss scaling
        use_amp = config["training"]["amp"] and self.device.type == "cuda"
        scaler = GradScaler(enabled=use_amp)

        # Repeatedly cycle over data by specified epochs

        start_epoch = self.config["training"]["start_epoch"]
        epochs = self.config["training"]["epochs"]
        best_loss = float("inf")

        for epoch_counter in range(start_epoch, epochs):

            running_loss = 0.0
            loop = tqdm(train_loader, desc=f"Epoch {epoch_counter}/{epochs}")

            for clips in loop:

                clips = clips.to(self.device, non_blocking=True)

                # Context manager: enable/disable AMP, i.e., dynamicly apply float16 precision or normal float32
                # Note: ignore pylance warning for `autocast, the docs havnt been updated to the depreciation
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.config["training"]["amp"],
                    dtype=torch.float16,
                ):
                    outputs = self.model(clips)
                    loss = self.criterion(
                        outputs, clips
                    )  # outputs PyTorch Tensor scalar: tesnor(float, device)

                self.optimizer.zero_grad(
                    set_to_none=True
                )  # zero out gradients to prevent accumulaion (PyTorch default is to accumulate)

                # compute grads and scale to prevent vanishing grads/underflow in float16 (AMP)
                scaler.scale(loss).backward()

                # if AMP: unscale grads, then update encoder params
                scaler.step(self.optimizer)

                scaler.update()  # Check if grads valid, adjust scaler dynamiclly (noIssues=increase, NaNs/Inf=decrease)

                # extract float only, avoid pytorch tensor accumulation (gradients, etc)
                running_loss += loss.item()

                # disp
                loop.set_postfix(loss=loss.item())

            avg_epoch_loss = running_loss / len(train_loader)

            # warmup period (scheduler holds learning rate constant for first 'args.warmup_epochs' epochs)
            if (epoch_counter) >= self.config["training"]["warmup_epochs"]:
                self.scheduler.step()

            # Writter logging:

            # Log scalars
            self.writer.add_scalar("AvgLoss/train", avg_epoch_loss, epoch_counter)
            self.writer.add_scalar(
                "learning_rate", self.scheduler.get_last_lr()[0], epoch_counter
            )

            # Additional options as pipeline developed:
            # =====================
            # If data split to train/val, may not be needed for this product demo
            # self.writer.add_scalar("Loss/val", val_loss, epoch_counter)

            # Post training anomaly_score matrics -> would need to pre-label clips with anomalies to use
            # self.writer.add_scalar("Anomaly/Precision", precision, epoch_counter)
            # self.writer.add_scalar("Anomaly/Recall", recall, epoch_counter)
            # self.writer.add_scalar("Anomaly/F1", f1, epoch_counter)

            # Log Reconstruction error stats
            # self.writer.add_scalar("ReconError/Mean", recon_errors.mean(), epoch_counter)
            # self.writer.add_scalar("ReconError/Std", recon_errors.std(), epoch_counter)
            # =====================

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss

                logging.save_checkpoint(
                    {
                        "best_train_loss": best_loss,
                        "epoch": epoch_counter,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                    },
                    os.path.join(
                        self.log_dir,
                        "best_model",
                        f"best_model.pth.tar",
                    ),
                )

            if epoch_counter % self.config["training"]["checkpoint_interval"] == 0:
                # Writer logs and checkpoint

                # Extract example batch on cpu then extract single frame from closest to middle of clip
                first_clip = next(iter(train_loader))[0]  # (C, T, H, W)
                middle_idx = first_clip.shape[1] // 2
                middle_frame = first_clip[:, middle_idx, :, :]  # (C, H, W)

                # self.writer.add_video("Reconstruction/sample", reconstructed_clip_tensor, epoch_counter, fps=5)
                self.writer.add_images(
                    "Input/sample", middle_frame.unsqueeze(0), epoch_counter
                )

                logging.save_checkpoint(
                    {
                        "best_train_loss": best_loss,
                        # "best_val_loss": best_val_loss,
                        "epoch": epoch_counter,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                    },
                    os.path.join(
                        self.log_dir,
                        "checkpoints",
                        f"checkpoint_epoch_{epoch_counter}.pth.tar",
                    ),
                )

        self.writer.close()
