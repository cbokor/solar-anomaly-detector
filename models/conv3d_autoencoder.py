# %% Imports

import torch
import torch.nn as nn

# %% Classes


class Basic3DAE(nn.Module):
    """
    A simple 3D convolutional autoencoder for video data, suitable for reconstructive anomaly detection tasks.

    adv:
    - Lightweight and efficient (faster training, lower VRAM).
    - Seperates downsampling from Conv3D via MAxPool3D,allowing explicit control over spatial/temporal compression ratio.
    - Better for datasets where the temporal dimension is short and spatial features are dominant.

    disadv:
    - Lower Capacity channels, so may not capture complex spatiotemporal features as well as deeper models.
    """

    def __init__(self):
        """
        Initialize the Conv3D Autoencoder.

        Notes:
        - padding = (kernel_size-1)//2. Used to preserve spatial/temporal resolution assuming stride=1.
        - conv_output_size = floor((input_size + 2*padding - kernel_size) / stride + 1)
        - Downsampling is done via MaxPool3D layers rather than strided convolutions (i.e., stride > 1)
        to maintain more spatial information.
        - In MaxPool3D, stride == kernel_size by default, downsampling the input by the kernel size.
        - Upsampling is then done via stride > 1 in ConvTranspose3D, reversing MaxPool3D symmetrically.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            # Note: nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0)
            nn.Conv3d(1, 16, kernel_size=3, padding=1),  # (3x3x3) kernel
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                (1, 2, 2)
            ),  # downsample (1, T, H, W) -> (1, T, H/2, W/2), spatial halved
            nn.Conv3d(
                16, 32, kernel_size=3, padding=1
            ),  # double the channels for greature feature complexity
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                (2, 2, 2)
            ),  # downsample (1, T, H/2, W/2) -> (1, T/2, H/4, W/4), spatial and temporal halved
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                32, 16, 2, stride=2
            ),  # upsample (1, T/2, H/4, W/4) -> (1, T, H/2, W/2), reverse MaxPool3D((2,2,2))
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(
                16, 1, (1, 2, 2), stride=(1, 2, 2)
            ),  # upsample (1, T, H/2, W/2) -> (1, T, H, W), reverse MaxPool3D((1,2,2))
            nn.Sigmoid(),  # Sigmoid to ensure output in range [0, 1] for pixel values (assume normalized input)
        )

    def forward(self, x):
        z = self.encoder(x)  # latent space z
        decoded = self.decoder(z)  # reconstructed output
        return decoded


class PaperConv3DAE(nn.Module):
    """
    A reconstructive 3D convolutional autoencoder to replicate the architecture from the paper
    “Detecting spatiotemporal irregularities in videos using 3D Convolutional Autoencoder”
    (Yokoyama & Nakazawa, 2019, JVCIR) -> http://dx.doi.org/10.1016/j.jvcir.2019.102747 .

    adv:
    - High capacity model for more complex spatiotemporal feature learning.
    - having stride manage downsampling removes need for pooling layers, increased layer efficiency.
    - Kernel=4 with stride=2 is known for reducing checkerboard artifacts in upsampling,
        often seen with deconvolutions. Produces better-looking reconstructions in video/image tasks.

    disadv:
    - Heavier model, 256 channels easily consumes multiple GB of VRAM, may not fit small GPUs for
        larger inputs/batch-sizes.
    - Regid structure, less flexibility in downsampling/upsampling ratios.
    - Aggresive downsampling may lose fine-grained spatial details, especially in small input videos.
    """

    def __init__(self):
        """
        Initialize the Conv3D Autoencoder.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 256, 4, stride=2, padding=1),  # 112→56
            nn.BatchNorm3d(256),
            nn.Tanh(),
            nn.Conv3d(256, 128, 4, stride=2, padding=1),  # 56→28
            nn.BatchNorm3d(128),
            nn.Tanh(),
            nn.Conv3d(128, 64, 4, stride=2, padding=1),  # 28→14
            nn.BatchNorm3d(64),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 4, stride=2, padding=1),  # 14→28
            nn.BatchNorm3d(64),
            nn.Tanh(),
            nn.ConvTranspose3d(64, 128, 4, stride=2, padding=1),  # 28→56
            nn.BatchNorm3d(128),
            nn.Tanh(),
            nn.ConvTranspose3d(128, 256, 4, stride=2, padding=1),  # 56→112
            nn.BatchNorm3d(256),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)  # latent space z
        decoded = self.decoder(z)  # reconstructed output
        return decoded
