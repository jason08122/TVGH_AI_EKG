import time
import os
import neurokit2 as nk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from models.resnet_simclr import ResNet1D18SimCLR


class LVSDClassifier:
    def __init__(self, checkpoint, device='cpu'):
        self.labels = ['Normal', 'Systolic Dysfunction']
        self.device = device
        self.model = ResNet1D18SimCLR(in_c=12).to(self.device)
        checkpoint = torch.load(checkpoint, map_location=self.device)['state_dict']
        in_dim = self.model.backbone.fc[0].in_features
        self.model.backbone.fc = nn.Sequential(nn.Linear(in_dim, 2)).to(self.device)
        self.model.load_state_dict(checkpoint)
        
    
    def preprocess(self, ecgDatas):
        '''
        Apply a high-pass 5-order butterworth filter with lowcut 0.5Hz
        Drop the first and the last second

        Input:
            ecgDatas: [patient_num, 12, 5000]
        Output:
            filteredEcgDatas: [patient_num, 12, 5000]
        '''
        filteredEcgDatas = []
        for ecgData in ecgDatas:
            assert ecgData.shape == (12, 5000), 'data per person must be (12, 5000)'
            filteredEcgData = []
            for ecglead in ecgData:
                ecglead = nk.signal_filter(signal=ecglead, sampling_rate=500, lowcut=0.5, method='butterworth', order=5)
                ecglead = nk.signal_filter(signal=ecglead, sampling_rate=500, method='powerline')
                filteredEcgData.append(ecglead)

            filteredEcgDatas.append(filteredEcgData)
        
        filteredEcgDatas = np.stack(filteredEcgDatas, dtype=np.float32)
        return filteredEcgDatas
    
    def predict(self, ecgDatas):
        '''
        Predict LVSD with 12-lead ECG input

        Input:
            ecgDatas:
                [patient_num, 12, 5000]
        Output:
            model_pred: [patient_num]
                0: Normal
                1: Systolic Dysfunction
        '''

        # ecgDatas = self.preprocess(ecgDatas)
        
        dataset = TensorDataset(torch.from_numpy(ecgDatas))
        dataloader = DataLoader(dataset, batch_size=64)
        
        model_pred = []
        self.model.eval()
        with torch.no_grad():
            for X in dataloader:
                X = X[0].to(self.device)
                pred, _ = self.model(X)
                model_pred.append(pred.argmax(1))

        probabilities = F.softmax(pred, dim=1)
        probabilities = probabilities.squeeze().tolist()
        model_pred = torch.cat(model_pred).cpu().numpy()
        model_pred = [self.labels[p] for p in model_pred]
        pred_diagnos = 'Normal' if model_pred[0] == 'Normal' else  'HFrEF'
        return probabilities, pred_diagnos
    

if __name__ == '__main__':
    lvsd = LVSDClassifier('./weights/checkpoint_0104.pth.tar')
    data = np.random.rand(1, 12, 5000).astype(np.float32)
    predict = lvsd.predict(data)
    print(predict)
