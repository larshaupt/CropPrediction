import torch
from src.utils import label_mask_to_onehot



class BCEMILLoss(torch.nn.Module):
    def __init__(self, n_classes, **kwargs):
        super(BCEMILLoss, self).__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss(**kwargs)
        self.n_classes = n_classes

    def forward(self, input, target):
        target = label_mask_to_onehot(target, self.n_classes)
        return self.bce_loss(input, target), target

