import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from einops import rearrange
import torch.nn.functional as F


class ModifiedEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedEfficientNet, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)

    def forward(self, x):
        x = x.tensors
        B, L, C, H, W = x.shape
        x = rearrange(x, 'B L C H W -> (B L) C H W')
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.model(x)
        x = x.view(B, L, -1)
        return x
