import torch


def label_mask_to_onehot(label_mask: torch.Tensor, n_classes: int):
    """
    Convert a label mask to one hot encoding.
    
    Parameters:
        - label_mask (torch.Tensor): Label mask tensor of shape [n_batch, height, width].
        - n_classes (int): Number of classes for one-hot encoding.
    
    Returns:
        - onehot (torch.Tensor): One hot encoding of shape [n_batch, n_classes].
    """
    if label_mask.dim() == 2:
        label_mask = label_mask.unsqueeze(0)

    # Flatten the label mask to shape [n_batch, n_pixels] where n_pixels = height * width
    n_batch = label_mask.size(0)
    label_mask_flat = label_mask.view(n_batch, -1)

    # Get the unique labels in each sample
    unique_labels = label_mask_flat.unique(dim=1)

    # Initialize a tensor for the one-hot encoding with zeros
    onehot = torch.zeros(n_batch, n_classes, device=label_mask.device)

    # Use advanced indexing to set 1 for each label present in the unique labels of each sample
    onehot.scatter_(1, unique_labels.to(torch.int64), 1).to(torch.uint8)

    return onehot

