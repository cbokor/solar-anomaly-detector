# %% Imports

import os
import torch
from torch.utils.data import Dataset


# %% Class


class ClipDataSet(Dataset):
    """
    Class to handle loading and transforming clips from a directory into PyTorch's Dataset API.
    Assumes clips are stored as .pt files in the specified directory.
    Each clip is expected to be a tensor, typically of shape (C, T, H, W).

    Note: If transforms include spatial transforms (like RandomCrop, RandomHorizontalFlip),
    they must be applied identically across all frames in a clip, otherwise breaking temporal consistency.
    Standard torchvision transforms dont handle temporal consistency out of the box.
    """

    def __init__(self, clip_dir, transform=None):
        """
        Initialize dataset with directory and optional transformations.
        """
        self.clip_dir = clip_dir
        self.transform = transform
        # extract only .pt files from given clip_dir
        self.clip_files = sorted(f for f in os.listdir(clip_dir) if f.endswith(".pt"))

    def __len__(self):
        """
        Return the number of clips in the dataset.
        """
        return len(self.clip_files)

    def __getitem__(self, idx):
        """
        Load a clip by index, apply transformations if any.
        """
        clip_path = os.path.join(self.clip_dir, self.clip_files[idx])

        # Security risk raised for torch.load(), specifically malicous unpickling, weights_only=True set to match future default (03/07/2025)
        # See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details
        clip = torch.load(clip_path, weights_only=True)

        if self.transform:
            clip = self.transform(clip)
        return clip
