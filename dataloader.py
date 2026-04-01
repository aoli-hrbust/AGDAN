from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import h5py

class Caltech():
    def __init__(self, path):
        # scaler = MinMaxScaler()
        data = scipy.io.loadmat(path + 'Caltech101-20.mat')
        self.Y = data['Y'].astype(np.int32).reshape(2386,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
        self.V4 = data['X'][0][3].astype(np.float32)
        self.V5 = data['X'][0][4].astype(np.float32)
        self.V6 = data['X'][0][5].astype(np.float32)
        self.X = [self.V1, self.V2, self.V3, self.V4, self.V5, self.V6]
    def __len__(self):
        return 2386
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3),
                torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Animal():
    def __init__(self, path):
        # scaler = MinMaxScaler()
        data = scipy.io.loadmat(path + 'Animal-50.mat')
        self.Y = data['gt'].astype(np.int32).reshape(10158,)
        self.V1 = data['X'][0][0].T.astype(np.float32)
        self.V2 = data['X'][0][1].T.astype(np.float32)
        self.X = [self.V1, self.V2]

    def __len__(self):
        return 10158
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class LandUse():
    def __init__(self, path):
        # scaler = MinMaxScaler()
        data = scipy.io.loadmat(path + 'LandUse-21.mat')
        self.Y = data['Y'].astype(np.int32).reshape(2100,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
        self.X = [self.V1, self.V2, self.V3]
        # self.V1 = scaler.fit_transform(data['X'][0][0].T.astype(np.float32))
        # self.V2 = scaler.fit_transform(data['X'][0][1].T.astype(np.float32))
        # self.V3 = scaler.fit_transform(data['X'][0][2].T.astype(np.float32))
    def __len__(self):
        return 2100
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class COIL():
    def __init__(self, path):
        # scaler = MinMaxScaler()
        data = scipy.io.loadmat(path + 'COIL-20.mat')
        self.Y = data['truth'].astype(np.int32).reshape(1440,)
        self.V1 = data['X'][0][0].T.astype(np.float32)
        self.V2 = data['X'][0][1].T.astype(np.float32)
        self.V3 = data['X'][0][2].T.astype(np.float32)
        self.X = [self.V1, self.V2, self.V3]
        # self.V1 = scaler.fit_transform(data['X'][0][0].T.astype(np.float32))
        # self.V2 = scaler.fit_transform(data['X'][0][1].T.astype(np.float32))
        # self.V3 = scaler.fit_transform(data['X'][0][2].T.astype(np.float32))
    def __len__(self):
        return 1440
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class MSRC():
    def __init__(self, path):
        # scaler = MinMaxScaler()
        data = scipy.io.loadmat(path + 'MSRCv1.mat')
        self.Y = data['gnd'].astype(np.int32).reshape(210,)
        self.V1 = data['X'][0][0].T.astype(np.float32)
        self.V2 = data['X'][0][1].T.astype(np.float32)
        self.V3 = data['X'][0][2].T.astype(np.float32)
        self.V4 = data['X'][0][3].T.astype(np.float32)
        self.V5 = data['X'][0][4].T.astype(np.float32)
        self.V6 = data['X'][0][5].T.astype(np.float32)
        self.X = [self.V1, self.V2, self.V3, self.V4, self.V5, self.V6]
        # self.X = [self.V1, self.V2, self.V3, self.V4, self.V5]
        # self.V1 = scaler.fit_transform(data['X'][0][0].T.astype(np.float32))
        # self.V2 = scaler.fit_transform(data['X'][0][1].T.astype(np.float32))
        # self.V3 = scaler.fit_transform(data['X'][0][2].T.astype(np.float32))
    def __len__(self):
        return 210
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class HW():
    def __init__(self, path):
        scaler = MinMaxScaler()
        data = scipy.io.loadmat(path + 'HW.mat')
        self.Y = data['Y'][0].astype(np.int32).reshape(2000,)
        self.V1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.V4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.V5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.V6 = scaler.fit_transform(data['X6'].astype(np.float32))
        self.X = [self.V1, self.V2, self.V3, self.V4, self.V5, self.V6]
    def __len__(self):
        return 2000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4),
                torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class USPS():
    def __init__(self, path):
        # scaler = MinMaxScaler()
        data = scipy.io.loadmat(path + 'USPS-MNIST.mat')
        self.Y = data['truth'].astype(np.int32).reshape(10000,)
        self.V1 = data['X'][0][0].T.astype(np.float32)
        self.V2 = data['X'][0][1].T.astype(np.float32)
        self.X = [self.V1, self.V2]
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class ORL():
    def __init__(self, path):
        # scaler = MinMaxScaler()
        data = scipy.io.loadmat(path + 'ORL-40.mat')
        self.Y = data['gt'].astype(np.int32).reshape(400,)
        self.V1 = data['X'][0][0].T.astype(np.float32)
        self.V2 = data['X'][0][1].T.astype(np.float32)
        self.V3 = data['X'][0][2].T.astype(np.float32)
        self.X = [self.V1, self.V2, self.V3]
    def __len__(self):
        return 400
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Scene():
    def __init__(self, path):
        # scaler = MinMaxScaler()
        data = scipy.io.loadmat(path + 'Scene-15.mat')
        self.Y = data['Y'].astype(np.int32).reshape(4485,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
        self.X = [self.V1, self.V2, self.V3]
    def __len__(self):
        return 4485
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

def load_data(dataset):
    if dataset == "Caltech101-20":
        dataset = Caltech('./data/')
        dims = [48, 40, 254, 1984, 512, 928]
        view = 6
        data_size = 2386
        class_num = 20
        labels = dataset.Y
    elif dataset == "Animal-50":
        dataset = Animal('./data/')
        dims = [4096, 4096]
        view = 2
        data_size = 10158
        class_num = 50
        labels = dataset.Y
    elif dataset == "LandUse-21":
        dataset = LandUse('./data/')
        dims = [20, 59, 40]
        view = 3
        data_size = 2100
        class_num = 21
        labels = dataset.Y
    elif dataset == "COIL-20":
        dataset = COIL('./data/')
        dims = [1024, 944, 576]
        view = 3
        data_size = 1440
        class_num = 20
        labels = dataset.Y
    elif dataset == "USPS-MNIST":
        dataset = USPS('./data/')
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
        labels = dataset.Y
    elif dataset == "HW":
        dataset = HW('./data/')
        dims = [216, 76, 64, 6, 240, 47]
        view = 6
        data_size = 2000
        class_num = 10
        labels = dataset.Y
    elif dataset == "MSRCv1":
        dataset = MSRC('./data/')
        dims = [1302, 48, 512, 100, 256, 200]
        view = 6
        data_size = 210
        class_num = 7
        labels = dataset.Y
    elif dataset == "ORL-40":
        dataset = ORL('./data/')
        dims = [4096, 3304, 6750]
        view = 3
        data_size = 400
        class_num = 40
        labels = dataset.Y
    elif dataset == "Scene-15":
        dataset = Scene('./data/')
        dims = [20, 59, 40]
        view = 3
        data_size = 4485
        class_num = 15
        labels = dataset.Y
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num, labels
