import torch
import pytest
from src.utils import label_mask_to_onehot
from src.losses import BCEMILLoss


@pytest.fixture
def setup_bce_mi_loss():
    batch_size = 4
    height, width = 244, 244
    n_classes = 3
    input = torch.randn(batch_size, n_classes).sigmoid()  # Logits passed through sigmoid
    target = torch.randint(0, n_classes, (batch_size, height, width), dtype=torch.uint8)  # Random class labels
    loss_fn = BCEMILLoss(n_classes=n_classes)
    return input, target, loss_fn

def test_bce_mi_loss(setup_bce_mi_loss):
    input, target, loss_fn = setup_bce_mi_loss

    # Compute the loss
    loss = loss_fn(input, target)

    # Assert the loss is a tensor and scalar
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor."
    assert loss.dim() == 0, "Loss should be a scalar."

    # Check the shape of the one-hot encoded target
    onehot_target = label_mask_to_onehot(target, n_classes=3)
    assert onehot_target.ndim == 2, "One-hot target should have 2 dimensions."
    assert onehot_target.shape[0] == input.shape[0], "Batch dimension should match."
    assert onehot_target.shape[1] == input.shape[1], "Number of classes should match."

    # Check if the loss is a valid number (not NaN or inf)
    assert not torch.isnan(loss), "Loss is NaN."
    assert not torch.isinf(loss), "Loss is infinite."

    print("Test passed successfully!")
