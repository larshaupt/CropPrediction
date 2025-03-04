import torch
from src.utils import label_mask_to_onehot


import torch

class BCEMILLoss(torch.nn.Module):
    """
    A custom loss function for multi-class classification with Binary Cross-Entropy loss and class balancing.

    Args:
        n_classes (int): The number of classes in the classification problem.
        merge_intercropping (bool): Whether to merge certain classes related to intercropping. Default is False.
        class_distribution (tensor, optional): A tensor representing the distribution of classes in the dataset.
        **kwargs: Additional arguments for BCEWithLogitsLoss.

    Attributes:
        n_classes (int): Number of classes.
        merge_intercropping (bool): Flag to merge intercropping classes.
        class_distribution (tensor): Distribution of classes in the dataset.
        class_weights (tensor): Weights for each class based on distribution.
    """
    
    def __init__(self, n_classes, merge_intercropping=False, class_distribution=None, device="cpu", **kwargs):
        """
        Initializes the BCEMILLoss class.

        Args:
            n_classes (int): Number of classes.
            merge_intercropping (bool): Whether to merge intercropping classes.
            class_distribution (tensor, optional): Class distribution to compute class weights.
            **kwargs: Additional arguments for BCEWithLogitsLoss.
        """
        super(BCEMILLoss, self).__init__()
        
        self.n_classes = n_classes
        self.merge_intercropping = merge_intercropping
        self.class_distribution = class_distribution
        self.class_weights = torch.ones(n_classes)  # Initialize class weights as ones
        self.device = device
        
        # If class distribution is provided, calculate the class weights
        if class_distribution is not None:
            self.set_class_weights()
            
        self.bce_loss = torch.nn.BCEWithLogitsLoss(weight=self.class_weights, **kwargs)  # Binary Cross-Entropy loss with logits

    def forward(self, input, target):
        """
        Forward pass of the loss function.

        Args:
            input (tensor): Predicted logits for each class.
            target (tensor): Ground truth labels, which are converted to one-hot encoding.

        Returns:
            tuple: The BCE loss value and the target in one-hot format.
        """
        # Convert target labels to one-hot encoding based on class number
        target = label_mask_to_onehot(target, self.n_classes, merge_intercropping=self.merge_intercropping)
        
        # Calculate BCE loss with the class weights
        return self.bce_loss(input, target.float()), target

    def set_class_weights(self):
        """
        Set class weights based on class distribution to address class imbalance.

        The weights are calculated as the inverse of the class distribution, and can be adjusted
        to account for merged classes (e.g., intercropping).
        """
        assert self.class_distribution is not None, "Class distribution must be provided"
        
        # Copy the class distribution to avoid modifying the original
        class_distribution = self.class_distribution

        # Merge intercropping classes if specified
        if self.merge_intercropping:
            class_distribution[1] = class_distribution[1] + class_distribution[4] + class_distribution[5] + class_distribution[6]  # Maize
            class_distribution[2] = class_distribution[2] + class_distribution[5]  # Cassava
            class_distribution[3] = class_distribution[3] + class_distribution[4]  # Combined cassava and maize
            class_distribution[4] = class_distribution[6]  # Adjusted cassava
            class_distribution = class_distribution[:5]  # Consider only the first 5 classes
            
        # Clip to avoid division by zero (set small values to the minimum non-zero class distribution)
        class_distribution = torch.clip(class_distribution, torch.min(class_distribution[class_distribution > 0]), None)
        
        # Calculate class weights as the inverse of class distribution
        sum_classes = torch.sum(class_distribution)
        class_weights = sum_classes / class_distribution
        
        # Assign the calculated weights to class_weights attribute
        self.class_weights = class_weights.to(self.device)


