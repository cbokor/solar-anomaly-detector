# %% Import

import torch.nn as nn
from pytorch_msssim import ssim

# %% Custom Loss Classes


class ExampleCustomLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # custom loss logic here
        pass


class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, input, target):
        input = input.float()
        target = target.float()
        return 1 - ssim(input, target, data_range=self.data_range)


class MSE_SSIMLoss(nn.Module):
    def __init__(self, alpha=0.5, data_range=1.0):
        super().__init__()
        self.alpha = alpha
        self.data_range = data_range
        self.mse = nn.MSELoss()

    def forward(self, input, target):
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
