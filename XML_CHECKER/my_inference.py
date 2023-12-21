import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np

from ecg_dataset import SingleDataDataset, AugCompose, FixedCropECG
from models.resnet_simclr import ResNet1D18SimCLR


def ai_test(ecg: np.array) -> list :
    device = torch.device('cpu')
    out_dim = 128
    num_classes = 3
    pretrainedFile = 'weights/super.tar'
    transform=AugCompose([FixedCropECG()])
    model = ResNet1D18SimCLR(out_dim=out_dim).to(device)

    in_dim = model.backbone.fc[0].in_features
    model.backbone.fc = nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, num_classes)
    ).to(device)

    checkpoint = torch.load(pretrainedFile, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    ecgdata = ecg.astype(np.float32)
    ecgdata = transform(ecgdata)
    single_data_dataset = SingleDataDataset(ecgdata)
    data_loader = DataLoader(single_data_dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for batch in data_loader:
            pred, _ = model(batch)
    
    probabilities = F.softmax(pred, dim=1)
    probabilities = probabilities.squeeze().tolist()

    return probabilities, pred.argmax(1).item()
    # return pred.argmax(1).item()
            

if __name__ == '__main__':

    ecg = np.load('test.npy')

    res = ai_test(ecg)
    print(res)