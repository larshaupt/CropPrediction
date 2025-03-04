import torch


def label_mask_to_onehot(label_mask: torch.Tensor, n_classes: int, merge_intercropping: bool = False) -> torch.Tensor:
    """
    Convert a label mask to one hot encoding.
    
    Parameters:
        - label_mask (torch.Tensor): Label mask tensor of shape [n_batch, height, width].
        - n_classes (int): Number of classes for one-hot encoding.
        - merge_intercropping (bool): Whether to merge intercropping classes (4,5,6,7)
    
    Returns:
        - onehot (torch.Tensor): One hot encoding of shape [n_batch, n_classes].
    """
    if label_mask.dim() == 2:
        label_mask = label_mask.unsqueeze(0)
        
    if merge_intercropping:
        n_classes = n_classes + 3

    # Flatten the label mask to shape [n_batch, n_pixels] where n_pixels = height * width
    n_batch = label_mask.size(0)
    label_mask_flat = label_mask.view(n_batch, -1)

    # Get the unique labels in each sample
    unique_labels = label_mask_flat.unique(dim=1)

    # Initialize a tensor for the one-hot encoding with zeros
    onehot = torch.zeros(n_batch, n_classes, device=label_mask.device)

    # Use advanced indexing to set 1 for each label present in the unique labels of each sample
    onehot = onehot.scatter_(1, unique_labels.to(torch.int64), 1).to(torch.uint8)
    
    if merge_intercropping:
        onehot[:,1] = onehot[:,1] | onehot[:,4] | onehot[:,5] | onehot[:,6] # maize
        onehot[:,2] = onehot[:,2] | onehot[:,5] # cassava
        onehot[:,3] = onehot[:,3] | onehot[:,4] # common bean
        onehot[:,4] = onehot[:,6]  #soybean
        onehot = onehot[:, :5] # remove extra classes

    return onehot

