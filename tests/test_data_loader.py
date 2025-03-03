import pytest
import torch
from src.data_loader import CropDataset, create_dataloader, get_transforms

# Define test paths (use dummy paths or mock)
TEST_ROOT = "../data/raw"
TEST_SPLIT_CSV = "../data/raw/FieldIds.csv"

@pytest.fixture
def dataset():
    """Fixture to initialize the dataset."""
    return CropDataset(root=TEST_ROOT, split_csv=TEST_SPLIT_CSV, chip_size=224, stride=16)

@pytest.fixture
def dataloader(dataset):
    """Fixture to create a dataloader."""
    return create_dataloader(root=TEST_ROOT, split_csv=TEST_SPLIT_CSV, batch_size=4, chip_size=224, stride=16, is_train=True, num_workers=0)

def test_dataset_length(dataset):
    """Test if dataset is loading properly."""
    assert len(dataset) > 0, "Dataset should not be empty."

def test_dataset_sample(dataset):
    """Test if dataset returns samples in correct format."""
    sample = dataset[0]
    assert "image" in sample, "Sample must contain 'image'."
    assert "field_ids" in sample, "Sample must contain 'field_ids'."
    assert "mask" in sample, "Sample must contain 'mask'."
    assert "tile_index" in sample, "Sample must contain 'tile_index'."
    assert isinstance(sample["image"], torch.Tensor), "Image must be a torch.Tensor."

def test_dataloader(dataloader):
    """Test if DataLoader is working correctly."""
    batch = next(iter(dataloader))
    assert "image" in batch, "Batch must contain 'image'."
    assert batch["image"].shape[0] == 4, "Batch size must match requested size."

def test_transforms():
    """
    Test if the transformations (histogram equalization and normalization) work correctly.
    """

    # Create a dummy tensor with shape [n_timepoints=1, n_bands=13, height=5, width=5]
    # The last band (cloud probability) will be a random value between 0 and 100.
    sample_input = torch.randn(13, 13, 244, 244)  # Random input tensor (for testing)

    # Add random values for the first 12 bands, and random cloud probabilities in the last channel
    sample_input[:, :12, :, :] = torch.randn(13, 12, 244, 244) * 1000  # Simulate spectral bands
    sample_input[:, 12, :, :] = torch.randint(0, 101, (13, 244, 244), dtype=torch.float32)  # Cloud probability in [0, 100]

    # Apply the transformations
    transform = get_transforms()
    transformed_sample = transform(sample_input)

    # Check if cloud probability (last channel) is divided by 100
    cloud_prob_transformed = transformed_sample[:, 12, :, :]
    assert torch.allclose(cloud_prob_transformed, sample_input[:, 12, :, :] / 100), "Cloud probability transformation failed."

    # Check min-max normalization 
    normalized_bands = transformed_sample
    min_values = normalized_bands.amin(dim=(0, 2, 3), keepdim=False)  # Min along height and width
    max_values = normalized_bands.amax(dim=(0, 2, 3), keepdim=False)  # Max along height and width

    # Ensure the min-max normalization is within [0, 1]
    assert torch.all(min_values >= 0), f"Min value is less than 0: {min_values}"
    assert torch.all(max_values <= 1), f"Max value is greater than 1: {max_values}"

    print("Transformation test passed!")