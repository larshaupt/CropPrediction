import torch
from torchmetrics import Metric
from torchmetrics.classification import F1Score

class F1ScorePerClass(Metric):
    def __init__(self, class_idx, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.class_idx = class_idx
        self.f1_metric = F1Score(task="binary")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.f1_metric.update(preds[:, self.class_idx], target[:, self.class_idx])
    
    def compute(self):
        return self.f1_metric.compute()