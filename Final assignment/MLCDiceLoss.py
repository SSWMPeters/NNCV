import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, smooth=1.0):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] - raw logits
        targets: [B, H, W] - class indices (0 to C-1)
        """
        if self.ignore_index is not None:
            targets = targets.clone()
            targets[targets == self.ignore_index] = 0
        
        # Apply softmax to get class probabilities
        inputs = F.softmax(inputs, dim=1)

        # One-hot encode targets: [B, H, W] → [B, C, H, W]
        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Flatten
        inputs = inputs.contiguous().view(inputs.size(0), inputs.size(1), -1)
        targets_one_hot = targets_one_hot.contiguous().view(targets_one_hot.size(0), targets_one_hot.size(1), -1)

        intersection = (inputs * targets_one_hot).sum(dim=2)
        union = inputs.sum(dim=2) + targets_one_hot.sum(dim=2)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        # Optional: mask out ignore_index class
        # if self.ignore_index is not None:
        #     mask = torch.ones(num_classes, dtype=torch.bool)
        #     mask[self.ignore_index] = False
        #     dice = dice[:, mask]

        # Return average Dice loss over classes and batch
        return 1 - dice.mean()
