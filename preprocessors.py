import re
from pathlib import Path
import os
import ntpath
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset

import set_seed


class TorchDataset(Dataset):
    def __init__(self, datapath):
        super().__init__()
        self.data = np.load(datapath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = torch.from_numpy(self.data[i]).to(torch.float)
        x, y = sample[:-1], sample[-1]
        return {'features': x, 'targets': y}
    
    
def make_dataset(data_dir):
    dataset = TorchDataset(data_dir)
    return dataset


def make_datasets(root_dir, files):
    root_dir = Path(root_dir)
    datasets = dict()
    for data_type in files:
        data_dir = root_dir/data_type
        dataset = TorchDataset(data_dir)
        name = ntpath.basename(os.path.splitext(data_dir)[0])
        datasets[name] = dataset
    return datasets
    

def make_dataloader(data_dir='data', data_file='test.npy', is_train=False, batch_size=128, subset_len=None, sample_method=None):
    dataset = make_dataset(Path(data_dir)/data_file)
    if subset_len:
        assert subset_len <= len(dataset)
        if sample_method == 'random':
            dataset = torch.utils.data.Subset(dataset, random.sample(range(0, len(dataset)), subset_len))
        else:
            dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
            
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return dataloader


def make_dataloaders(data_dir='data', data_ext = '.npy', batch_size=128):
    dataloaders = {}
    for f in os.listdir(data_dir):
        if f.endswith(data_ext):
            name = os.path.splitext(f)[0]
            is_train = re.findall('train', name)
            dataloader = make_dataloader(data_dir, f, is_train, batch_size)
            dataloaders[name] = dataloader
    return dataloaders
    


def dataloader_from_dataset(dataset, batch_size, name, shuffle=False):
    return {name: DataLoader(dataset, batch_size, shuffle=shuffle)}


def dataloaders_from_datasets(datasets, batch_size):
    dataloaders = dict()
    for mode, dataset in datasets.items():
        dataloaders[mode] = DataLoader(dataset, batch_size, shuffle=True if mode=='train' else False)
    return dataloaders


def load_from_state_dict(model, state_dict_path):
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    return model


def get_model_size(model):
    torch.save(model.state_dict(), 'tmp.p')
    sz = os.path.getsize('tmp.p')
    os.remove('tmp.p')
    return sz
    
