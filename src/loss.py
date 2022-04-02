
from torch.nn import functional as F
from .metrics import dice_score
import torch

class SegmentationCriterion:

    def __init__(self, dice_loss_weight=0.5): 

        self.dice_loss_weight = 0.5

    def __call__(self, logits, target_mask):

        cross_entropy = F.cross_entropy(logits, target_mask)

        pred_mask = torch.argmax(logits, dim=1)

        dice_loss = -dice_score(pred_mask, target_mask)

        return cross_entropy * (1 - self.dice_loss_weight) + self.dice_loss_weight * dice_loss

