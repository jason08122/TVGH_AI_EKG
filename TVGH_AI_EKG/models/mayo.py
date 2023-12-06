import torch
import torch.nn as nn


class MayoCNNECG4(nn.Module):
    def __init__(self, in_c: int = 12, num_classes: int = 2):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding='same', bias=True),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),

            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding='same', bias=True),
            # nn.BatchNorm2d(num_features=32),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding='same', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 4), padding=0),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 5), stride=(1, 1), padding='same', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 4), padding=0),
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(in_c, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(78, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.6),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=0.6),

            nn.Linear(8, num_classes)
        )

    def forward(self, x): # [batch_size, 12, 2500] -> [batch_size, 39]
        x = torch.unsqueeze(x, 1)
        t = self.temporal(x)
        s = self.spatial(t)
        s = s.flatten(start_dim=1)
        o = self.fc(s)
        return o, s



class MayoCNNECG6(nn.Module):
    def __init__(self, in_c: int = 12, num_classes: int = 2):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding='same', bias=True),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding='same', bias=True),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding='same', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 5), stride=(1, 1), padding='same', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), stride=(1, 1), padding='same', bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 5), stride=(1, 1), padding='same', bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(in_c, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(39, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.6),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=0.6),

            nn.Linear(8, num_classes)
        )

    def forward(self, x): # [batch_size, 12, 2500] -> [batch_size, 39]
        x = torch.unsqueeze(x, 1)
        t = self.temporal(x)
        s = self.spatial(t)
        s = s.flatten(start_dim=1)
        o = self.fc(s)
        return o, s

if __name__ == '__main__':
    in_c = 12
    x = torch.randn(32, in_c, 2500)
    model = MayoCNNECG4(in_c=in_c)
    print(model)
    y, _ = model(x)
    print(y.shape)
