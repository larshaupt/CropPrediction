# src/data_loader.py

import os
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from skimage import exposure
import torch
from torchgeo.datasets import CV4AKenyaCropType
from torchvision.transforms import ToTensor


def histogram_equalization(x):
    """Applies histogram equalization to the first 12 bands, leaves the last band unchanged."""
    x_np = x.numpy()  # Convert to NumPy
    x_eq = torch.Tensor(np.array([[exposure.equalize_hist(x_np[t, i]) for i in range(12)] for t in range(x.shape[0])], dtype = np.float32))
    return torch.cat([x_eq, x[:, 12:] / 100], dim=1)  # Normalize cloud probability to [0,1]

def normalize_bands(x):
    """Normalizes the first 12 spectral bands, leaves cloud probability as is."""
    mean = torch.mean(x[:, :12], dim=(0, 2, 3), keepdim=True)
    std = torch.mean(x[:, :12], dim=(0, 2, 3), keepdim=True)
    min_values = mean - 2 * std
    max_values = mean + 2 * std
    norm_bands = (x[:, :12] - min_values) / (max_values - min_values)
    # clipping
    norm_bands = torch.clamp(norm_bands, 0, 1)
    return torch.cat([norm_bands, x[:, 12:]], dim=1)


# Update the transform function to use these statistics
def get_transforms():
    """
    Returns the default transformations to apply to the images.
    This can include normalization, resizing, etc.

    Returns:
        - transform (callable): Transformation function.
    """
    transform = T.Compose([
        T.Lambda(histogram_equalization),  # Apply histogram equalization
        T.Lambda(normalize_bands)  # Normalize first 12 bands
    ])
    return transform


class CropDataset(Dataset):
    """
    Custom dataset class for loading the CV4A Kenya Crop Type dataset.
    """
    def __init__(self, root, split_csv, transform=None, chip_size=224, stride=16):
        """
        Initializes the dataset.

        Parameters:
            - root (str): Path to the dataset root.
            - split_csv (str): Path to CSV file containing train/test split.
            - transform (callable, optional): A function/transform to apply to the data.
            - chip_size (int, optional): Size of image chips to extract.
            - stride (int, optional): Stride size for chip extraction.
        """
        self.root = Path(root)
        self.split = pd.read_csv(split_csv)
        self.train_ids = self.split["train"].dropna().values
        self.test_ids = self.split["test"].dropna().values
        self.transform = transform
        self.chip_size = chip_size
        self.stride = stride

        # Load the dataset using TorchGeo's CV4A dataset
        self.dataset = CV4AKenyaCropType(root=self.root, download=False, chip_size=self.chip_size, stride=self.stride)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Fetches an item from the dataset.

        Parameters:
            - idx (int): Index of the sample to fetch.
        
        Returns:
            - dict: A dictionary containing the image, field_ids, mask, and tile_index.
        """
        sample = self.dataset[idx]
        image = sample["image"]  # Image is in shape [n_timepoints, n_bands, height, width]
        field_ids = sample["field_ids"]
        mask = sample["mask"]
        tile_index = sample["tile_index"]

        # Apply any transformations (e.g., normalization, augmentations)
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "field_ids": field_ids,
            "mask": mask,
            "tile_index": tile_index
        }

def create_dataloader(root, split_csv, batch_size=16, chip_size=224, stride=16, is_train=True, num_workers=4):
    """
    Creates a DataLoader for the dataset.

    Parameters:
        - root (str): Path to the dataset root.
        - split_csv (str): Path to CSV file containing train/test split.
        - batch_size (int): Batch size for the DataLoader.
        - chip_size (int): Size of image chips to extract.
        - stride (int): Stride size for chip extraction.
        - is_train (bool): Whether to create a DataLoader for training or testing.

    Returns:
        - dataloader (DataLoader): The created DataLoader.
    """
    # TODO: Directly apply normalizations at initialization
    # TODO: Add more transformations (e.g., augmentations)
    transform = get_transforms()

    # Choose the correct subset based on whether we're training or testing
    dataset = CropDataset(
        root=root,
        split_csv=split_csv,
        transform=transform,
        chip_size=chip_size,
        stride=stride
    )

    # Filter for training or testing split
    # TODO: Speed this up by saving the indices of the training and testing samples
    if is_train:
        # Only load the training IDs
        train_ids = dataset.train_ids
        dataset.dataset = [item for item in dataset.dataset if item["tile_index"] in train_ids]
    else:
        # Only load the testing IDs
        test_ids = dataset.test_ids
        dataset.dataset = [item for item in dataset.dataset if item["tile_index"] in test_ids]

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

    return dataloader
