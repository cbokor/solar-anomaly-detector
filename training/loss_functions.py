# %% Import

import torch.nn as nn
from pytorch_msssim import ssim

# %% Custom Loss Classes


class ExampleCustomLoss(nn.Module):
    """
    Template for a custom loss function.
    Replace with actual logic in forward().
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        """
        Compute custom loss between input and target.

        Args:
            input (Tensor): Predicted output.
            target (Tensor): Ground truth.

        Returns:
            Tensor: Scalar loss value.
        """
        pass


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) based loss.
    Encourages perceptual similarity rather than pixel-wise fidelity.
    """

    def __init__(self, data_range=1.0):
        """
        Args:
            data_range (float): Maximum possible pixel value (e.g., 1.0 or 255).
        """
        super().__init__()
        self.data_range = data_range

    def forward(self, input, target):
        """
        Computes SSIM-based loss: 1 - SSIM.

        Args:
            input (Tensor): Predicted image.
            target (Tensor): Ground truth image.

        Returns:
            Tensor: Scalar SSIM loss.
        """

        input = input.float()
        target = target.float()
        return 1 - ssim(input, target, data_range=self.data_range)


class MSE_SSIMLoss(nn.Module):
    """
    Combined MSE and SSIM loss.
    Useful for balancing pixel-level accuracy with perceptual quality.
    """

    def __init__(self, alpha=0.5, data_range=1.0):
        """
        Args:
            alpha (float): Weighting factor between MSE and SSIM (0 = all SSIM, 1 = all MSE).
            data_range (float): Maximum possible pixel value (e.g., 1.0 or 255).
        """

        super().__init__()
        self.alpha = alpha
        self.data_range = data_range
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        """
        Compute weighted combination of MSE and SSIM losses.

        Args:
            input (Tensor): Predicted image.
            target (Tensor): Ground truth image.

        Returns:
            Tensor: Scalar loss value.
        """

        input = input.float()
        target = target.float()
        mse_loss = self.mse(input, target)
        ssim_loss = 1 - ssim(
            input, target, win_size=7, size_average=True, data_range=self.data_range
        )
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss


# %% Loss Registry

LOSS_REGISTRY = {
    "MSELoss": nn.MSELoss,
    "L1Loss": nn.L1Loss,
    "SmoothL1Loss": nn.SmoothL1Loss,
    "SSIMLoss": SSIMLoss,
    "MSE_SSIMLoss": MSE_SSIMLoss,
    "CustomLoss": ExampleCustomLoss,  # Replace with custom loss class
    # Add other loss functions here as needed
}

"""
Dictionary mapping string keys to loss classes.

This enables dynamic loss selection from config or CLI, e.g.:
    loss_fn = LOSS_REGISTRY["MSE_SSIMLoss"]()
"""
