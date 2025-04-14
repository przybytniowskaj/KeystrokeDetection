import torch.nn as nn
from torch.nn.modules.activation import GELU
from torch.nn.modules.pooling import AdaptiveAvgPool2d


class Residual(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn=fn
    def forward(self,x):
        return x+self.fn(x)


class ConvMixer(nn.Sequential):
    def __init__(self, dim, depth, kernel_size, patch_size, num_classes):
        super().__init__(
            nn.Sequential(
                nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
                nn.GELU(),
                nn.BatchNorm2d(dim),
                *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size//2),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                ) for _ in range(depth)],
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim, num_classes)
                )
        )