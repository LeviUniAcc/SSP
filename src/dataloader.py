from torch.utils.data import Dataset
import os
import pickle as pkl


class SSPDataset(Dataset):

    def __init__(self, path, type):
        self.type = type
        self.path = path + '/' + type

    def __len__(self):
        return -1

    def __getitem__(self, idx):
        img_name = os.path.join(self.path, f'environment_ssps_idx_{str(idx)}.pkl')
        with open(img_name, 'rb') as f:
            ssp = pkl.load(f)
            return ssp


if __name__ == "__main__":
    dataset = SSPDataset(path='data/external/bib_train/ssps', type='single_object')
    length = dataset.__len__()
    item = dataset.__getitem__(idx=3)
    pass
