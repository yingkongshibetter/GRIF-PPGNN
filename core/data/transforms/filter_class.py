import torch
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

import pickle
class FilterClassByCount(BaseTransform):
    def __init__(self, min_count: int, remove_unlabeled=False):
        self.min_count = min_count
        self.remove_unlabeled = remove_unlabeled

    def __call__(self, data: Data) -> Data:
        assert hasattr(data, 'y')
        if data.name=='Pokec':
            mask = torch.ones(66569, dtype=torch.bool)
            mask[torch.where(data.y == -1)] = False
        else:
            y: torch.Tensor = F.one_hot(data.y)
            counts = y.sum(dim=0)
            y = y[:, counts >= self.min_count]
            mask = y.sum(dim=1).bool()        # nodes to keep
            data.y = y.argmax(dim=1)

        if self.remove_unlabeled:
            data = data.subgraph(mask)
        else:
            if data.name=='Pokec':
                if hasattr(data, 'train_mask'):
                    data.train_mask = data.train_mask & mask
                    data.val_mask = data.val_mask & mask
                    data.test_mask = data.test_mask & mask
            else:

                data.y[~mask] = -1
                if hasattr(data, 'train_mask'):
                    data.train_mask = data.train_mask & mask
                    data.val_mask = data.val_mask & mask
                    data.test_mask = data.test_mask & mask
                # set filtered nodes as unlabeled


        return data