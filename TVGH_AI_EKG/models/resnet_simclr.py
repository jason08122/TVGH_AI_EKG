import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor
import math
import torch.functional as F
import torch.optim as optim

def conv10(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, kernel_size=11, stride=stride, padding=5, bias=False)

def conv3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MLP(nn.Module):
    def __init__(self, in_dim: int = 6, out_dim: int = 1) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_dim, 8), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(4, 1), nn.ReLU())
        self.text = []

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        self.text = x
        return x

class Bottleneck1D(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        
        self.conv2 = conv3(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)

        self.conv3 = conv1(planes, planes*self.expansion)
        self.bn3 = nn.BatchNorm1d(planes*self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

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

class ResNet1D50(nn.Module):
    def __init__(self, num_classes: int, in_c: int = 12):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv1d(in_c, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),  # same

            # ======= layer 1 =======
            Bottleneck1D(64, 64, 1, nn.Sequential(
                conv1(64, 256, 1),
                nn.BatchNorm1d(256)
            )),
            Bottleneck1D(256, 64, 1),
            Bottleneck1D(256, 64, 1),

            # ======= layer 2 =======
            Bottleneck1D(256, 128, 2, nn.Sequential(
                conv1(256, 512, 2),
                nn.BatchNorm1d(512)
            )),
            Bottleneck1D(512, 128, 1),
            Bottleneck1D(512, 128, 1),
            Bottleneck1D(512, 128, 1),

            # ======= layer 3 =======
            Bottleneck1D(512, 256, 2, nn.Sequential(
                conv1(512, 1024, 2),
                nn.BatchNorm1d(1024)
            )),
            Bottleneck1D(1024, 256, 1),
            Bottleneck1D(1024, 256, 1),
            Bottleneck1D(1024, 256, 1),
            Bottleneck1D(1024, 256, 1),
            Bottleneck1D(1024, 256, 1),

            # ======= layer 4 =======
            Bottleneck1D(1024, 512, 2, nn.Sequential(
                conv1(1024, 2048, 2),
                nn.BatchNorm1d(2048)
            )),
            Bottleneck1D(2048, 512, 1),
            Bottleneck1D(2048, 512, 1),


            nn.AdaptiveAvgPool1d(1)   # same
        )
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x:Tensor) -> Tensor:
        x = self.res(x)
        h = torch.flatten(x, 1)
        x = self.fc(h)
        return x, h

class ResNet1D34(nn.Module):
    def __init__(self, num_classes: int, in_c: int = 12):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv1d(in_c, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            # ======= layer 1 =======
            BasicBlock1D(64, 64, 1),
            BasicBlock1D(64, 64, 1),
            BasicBlock1D(64, 64, 1),

            # ======= layer 2 =======
            BasicBlock1D(64, 128, 2, nn.Sequential(
                conv1(64, 128, 2),
                nn.BatchNorm1d(128)
            )),
            BasicBlock1D(128, 128, 1),
            BasicBlock1D(128, 128, 1),
            BasicBlock1D(128, 128, 1),

            # ======= layer 3 =======
            BasicBlock1D(128, 256, 2, nn.Sequential(
                conv1(128, 256, 2),
                nn.BatchNorm1d(256)
            )),
            BasicBlock1D(256, 256, 1),
            BasicBlock1D(256, 256, 1),
            BasicBlock1D(256, 256, 1),
            BasicBlock1D(256, 256, 1),
            BasicBlock1D(256, 256, 1),

            # ======= layer 4 =======
            BasicBlock1D(256, 512, 2, nn.Sequential(
                conv1(256, 512, 2),
                nn.BatchNorm1d(512)
            )),
            BasicBlock1D(512, 512, 1),
            BasicBlock1D(512, 512, 1),

            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(512, num_classes)
        

    def forward(self, x:Tensor ) -> Tensor:
        x = self.res(x)
        h = torch.flatten(x, 1)
        x = self.fc(h)
        return x, h

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

    def forward(self, x):
        return self.backbone(x)

class ResNet1D50SimCLR(nn.Module):
    def __init__(self, out_dim: int = 128, in_c: int = 12):
        super().__init__()
        self.backbone = ResNet1D50(num_classes=out_dim, in_c=in_c)
        dim_mlp = self.backbone.fc.in_features
        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x):
        return self.backbone(x)

class ResNet1D34SimCLR(nn.Module):
    def __init__(self, out_dim: int = 128, in_c: int = 12):
        super().__init__()
        self.backbone = ResNet1D34(num_classes=out_dim, in_c=in_c)
        dim_mlp = self.backbone.fc.in_features
        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x):
        return self.backbone(x)

class ResNet1D18SimCLRTab(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.backbone = ResNet1D18SimCLR(out_dim=out_dim)
        self.mlp = MLP(in_dim=2)

    def forward(self, x, y):
        x = self.backbone.backbone.res(x)
        h = torch.flatten(x, 1)
        y = self.mlp(y)
        z = torch.cat((h, y), dim=1)
        z = self.backbone.backbone.fc(z)

        return z

class ResNet1D50SimCLRTab(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.backbone = ResNet1D50SimCLR(out_dim=out_dim)
        self.mlp = MLP(in_dim=2)

    def forward(self, x, y):
        x = self.backbone.backbone.res(x)
        h = torch.flatten(x, 1)
        y = self.mlp(y)
        z = torch.cat((h, y), dim=1)
        z = self.backbone.backbone.fc(z)

        return z

class ResNet1D34SimCLRTab(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.backbone = ResNet1D34SimCLR(out_dim=out_dim)
        self.mlp = MLP(in_dim=2)

    def forward(self, x, y):
        x = self.backbone.backbone.res(x)
        h = torch.flatten(x, 1)
        y = self.mlp(y)
        z = torch.cat((h, y), dim=1)
        z = self.backbone.backbone.fc(z)

        return z

class MLPSimCLRTab(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.backbone = MLP()
        self.fc = nn.Linear(out_dim,out_dim)

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)

if __name__ == "__main__":
    x = torch.rand(1,12,5000)
    # m = CNNmodule(1, 16, 3, 1, 1)
    # m = CNN1d_adaptive(3, 5, 1, 1, 1)
    model = ResNet1D34(num_classes=128)
    # print(f'Total Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # out = model(x)

    out, _ = model(x)
    print(f'Total Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(out.shape)
    

