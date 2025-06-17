import torch
from torch.utils.data import Dataset
import numpy as np

class BBoxDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        self.split = split
        self.size = 100  # placeholder size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        rgb = torch.randn(3, 224, 224)
        pc = torch.randn(1024, 3)
        mask = torch.randint(0, 2, (224, 224))
        gt = torch.tensor([0, 0, 0, 1, 1, 1, 0], dtype=torch.float32)  # dummy box
        return rgb, pc, mask, gt
