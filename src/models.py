import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Load standard ResNet-18
        self.model = models.resnet18(weights=None)

        # CIFAR-10 images are 32x32, so modify the stem:
        # - smaller kernel
        # - stride 1 instead of 2
        # - remove maxpool
        self.model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.model.maxpool = nn.Identity()

        # Final classifier
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
