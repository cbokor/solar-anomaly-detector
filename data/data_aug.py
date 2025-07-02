# %% Import

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
    For real 3D transforms, need something like:
        - torchvision.io.VideoReader + torchvision.transforms.videotransforms (limited)
        - third-party libraries like PyTorchVideo for video-clip-aware transforms.
        - write custom transform functions that apply the same random parameters to each frame in the clip.

    """

    def __init__(self, clip_dir, transform=None):
        """
        Initialize dataset with directory and optional transformations.
        """
        self.clip_dir = clip_dir
        self.transform = transform
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
        clip = torch.load(clip_path)
        if self.transform:
            clip = self.transform(clip)
        return clip
