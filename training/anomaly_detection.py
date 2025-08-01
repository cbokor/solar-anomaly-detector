# %% Imports

import torch
import utils.logging as logging
import os
from tqdm import tqdm
from datetime import datetime
from torch.amp.grad_scaler import GradScaler

# %% Class


class AnomalyDetection:
    """
    Wrapper class for managing the anomaly detection training pipeline.

    This class encapsulates all necessary components for training an anomaly detection model,
    including the model itself, the optimizer, learning rate scheduler, loss function, and
    logging via TensorBoard's SummaryWriter.

    Attributes:
        model (nn.Module): The PyTorch model used for anomaly detection.
        optimizer (torch.optim.Optimizer): Optimizer used to update model weights.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        criterion (Callable): Loss function used to train the model.
        config (dict): Configuration dictionary with training and model parameters.
        device (torch.device): Device to run the model on (CPU or CUDA).
        log_dir (str): Path to the directory where logs are saved.
        writer (SummaryWriter): TensorBoard SummaryWriter for logging metrics and outputs.
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
        """Load previous specified checkpoint into encoder, default: args.checkpoint_path"""

        # UNTESTED!

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
        Train the model using a reconstructive objective.

        This method executes the main training loop for a reconstructive anomaly detection model,
        optionally utilizing Automatic Mixed Precision (AMP) to speed up training and reduce memory usage.

        Logging is handled using TensorBoard SummaryWriter. Checkpoints are saved periodically
        and whenever a new best loss is achieved.

        Parameters
        ----------
            config : dict
                Configuration dictionary containing training parameters.
            train_loader : DataLoader
                PyTorch DataLoader providing the training video clips.
        """

        # Enable AMP if specified in config and supported by device
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

                # Zero out gradients to prevent accumulaion (faster with `set_to_none=True`)
                self.optimizer.zero_grad(set_to_none=True)

                # Compute grads and scale to prevent vanishing grads/underflow in float16 (AMP)
                scaler.scale(loss).backward()

                # If AMP: unscale grads, then update encoder params
                scaler.step(self.optimizer)

                scaler.update()  # Check if grads valid, adjust scaler dynamiclly (noIssues=increase, NaNs/Inf=decrease)

                # Accumulate loss, extract float only to avoid pytorch tensor accumulation (gradients, etc)
                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_epoch_loss = running_loss / len(train_loader)

            # Update learning rate scheduler (after warmup period)
            if (epoch_counter) >= self.config["training"]["warmup_epochs"]:
                self.scheduler.step()

            # Log scalar metrics to TensorBoard
            self.writer.add_scalar("AvgLoss/train", avg_epoch_loss, epoch_counter)
            self.writer.add_scalar(
                "learning_rate", self.scheduler.get_last_lr()[0], epoch_counter
            )

            # Save best-performing model
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

            # Save periodic checkpoints and example inputs
            if epoch_counter % self.config["training"]["checkpoint_interval"] == 0:

                # Extract a middle frame from the first clip batch for visualization
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

        # Finalize and close the writer
        self.writer.close()
