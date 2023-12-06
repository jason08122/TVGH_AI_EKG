from torch.utils.data import Dataset
import numpy as np
import random

class AugCompose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class RandomCropECG:
    def __init__(self, size=2500):
        self.size = size
    
    def __call__(self, x):
        r = np.random.randint(500, x.shape[-1]-self.size)
        return x[:, r:r+self.size]

class FixedCropECG:
    def __init__(self, size=2500):
        self.size = size
    
    def __call__(self, x):
        return x[:, :self.size]


class RandomFilterECG:
    def __init__(self):
        self.filter = ["neurokit", "pantompkins1985", "hamilton2002", "elgendi2010", "engzeemod2012"]
        self.prev = "xxx"
    def __call__(self, x):
        method = random.choice(self.filter)
        while method == self.prev:
            method = random.choice(self.filter)

        self.prev = method
        outputs = []
        for onelead in x:
            output = nk.ecg_clean(onelead, sampling_rate=500, method=method)
            outputs.append(output)
        
        outputs = np.vstack(outputs).astype(np.float32)
        return outputs

class SingleDataDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return 1  # Since there is only one data point

    def __getitem__(self, idx):
        return self.data

# if __name__ == '__main__':
#     dataset = TVGHPretrainedDataset(anno='./anno/pretrain.csv')
#     for i in tqdm(range(len(dataset))):
#         a = dataset[i]
#         print(a)

    # training_data = TVGHFinetunedDataset(
    #     f'anno/tr85-17w-tab/test.csv', transform=AugCompose([FixedCropECG()]))

    # train_dataloader = DataLoader(
    #     training_data, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)


    # for X, t, l in tqdm(train_dataloader, ncols=80):
    #     print(f'{X}\n{t}\n{l}')
    #     break
    
