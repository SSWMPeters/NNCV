import torch
import torch.nn as nn
from torchvision import models


class Model(nn.Module):
    def __init__(self, num_classes=19):
        super(Model, self).__init__()
        # self.vit = vit_model
        self.vit = models.vit_b_16(pretrained=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1000, 512, kernel_size=2, stride=2),  # 14 → 28
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),   # 28 → 56
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),   # 56 → 112
            nn.ReLU(),
            nn.ConvTranspose2d(128, num_classes, kernel_size=2, stride=2),  # 112 → 224
        )

    def forward(self, x):
        features = self.vit(x)
        features = features.unsqueeze(2).unsqueeze(3)
        features = features.expand(-1, 1000, 14, 14)
        output = self.decoder(features)
        return output