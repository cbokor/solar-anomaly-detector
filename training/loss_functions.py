# %% Import

import torch.nn as nn

# %% Custom Loss Classes


class ExampleCustomLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # custom loss logic here
        pass


# %% Loss Registry

LOSS_REGISTRY = {
    "MSELoss": nn.MSELoss,
    "CustomLoss": ExampleCustomLoss,  # Replace with actual custom loss class
    # Add other loss functions here as needed
}
