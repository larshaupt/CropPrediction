import torch
import pytest
from src.model import CropClassifierWithGRU  # Adjust the import according to your structure


def test_model_forward_pass():
    """
    Test the forward pass of the model with randomly generated data.
    """
    batch_size = 8
    time_steps = 13
    n_bands = 13
    height, width = 224, 224  # Example dimensions based on your image chips

    # Generate random data for the model input (e.g., Sentinel-2 image chips)
    dummy_input = torch.randn(batch_size, time_steps, n_bands, height, width)
    
    # Initialize the model
    model = CropClassifierWithGRU(num_classes=10)  # Adjust according to your number of classes
    
    # Set the model in evaluation mode (for inference)
    model.eval()

    # Perform a forward pass
    with torch.no_grad():  # No need to track gradients for the test
        output = model(dummy_input)

    # Check that the output shape matches the expected shape
    assert output.shape == (batch_size, 10), f"Expected output shape (batch_size, num_classes), but got {output.shape}"

def test_model_gradients():
    """
    Test that gradients are properly computed during the backward pass.
    """
    batch_size = 8
    time_steps = 13
    n_bands = 13
    height, width = 224, 224  # Example dimensions based on your image chips

    # Generate random data for the model input
    dummy_input = torch.randn(batch_size, time_steps, n_bands, height, width)
    dummy_input.requires_grad = True  # Enable gradient tracking

    # Initialize the model
    model = CropClassifierWithGRU(num_classes=10)  # Adjust according to your number of classes

    # Set the model in training mode
    model.train()

    # Perform a forward pass
    output = model(dummy_input)

    # Compute a dummy loss (e.g., mean of output)
    loss = output.mean()

    # Perform backward pass
    loss.backward()

    # Check if gradients are computed for the input tensor
    assert dummy_input.grad is not None, "Gradients were not computed for the input tensor."

    print("Gradient test passed!")

# Run the tests
if __name__ == "__main__":
    pytest.main()
