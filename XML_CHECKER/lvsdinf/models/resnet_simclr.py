import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor

def conv3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock1D(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet1D18(nn.Module):
    def __init__(self, num_classes: int, in_c: int = 12):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv1d(in_c, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            BasicBlock1D(64, 64, 1),
            BasicBlock1D(64, 64, 1),

            BasicBlock1D(64, 128, 2, nn.Sequential(
                conv1(64, 128, 2),
                nn.BatchNorm1d(128)
            )),
            BasicBlock1D(128, 128, 1),

            BasicBlock1D(128, 256, 2, nn.Sequential(
                conv1(128, 256, 2),
                nn.BatchNorm1d(256)
            )),
            BasicBlock1D(256, 256, 1),

            BasicBlock1D(256, 512, 2, nn.Sequential(
                conv1(256, 512, 2),
                nn.BatchNorm1d(512)
            )),
            BasicBlock1D(512, 512, 1),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x:Tensor ) -> Tensor:
        x = self.res(x)
        h = torch.flatten(x, 1)
        x = self.fc(h)
        return x, h


class ResNet1D18SimCLR(nn.Module):
    def __init__(self, out_dim: int = 128, in_c: int = 12):
        super().__init__()
        self.backbone = ResNet1D18(num_classes=out_dim, in_c=in_c)
        dim_mlp = self.backbone.fc.in_features
        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x): # [batch_size, 12, 2500] -> [batch_size, out_dim]
        return self.backbone(x)
