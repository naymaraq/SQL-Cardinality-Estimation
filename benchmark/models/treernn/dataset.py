import numpy as np
import torch
from torch.utils.data import Dataset

from benchmark.models.treernn.featurize import index


class GaugeDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, trees, labels, input_dim, min_y=None, max_y=None):

        self.trees = trees
        self.labels = labels
        self.traverses = [tree.traverse_by_values() for tree in self.trees]
        self.inputs = [torch.stack(
            [torch.tensor([0.] * input_dim) if p_ in ['|', "&"] else torch.tensor(index.featurize(p_)) for p_ in trv])
                       for trv in self.traverses]

        self.normalize_dataset(min_y, max_y)

        self.trees = dict(enumerate(self.trees))
        self.labels = dict(enumerate(self.labels))
        self.inputs = dict(enumerate(self.inputs))

    def normalize_dataset(self, min_y=None, max_y=None):

        y = [k + 1 for k in self.labels]
        y = np.log(y)

        self.min_y, self.max_y = y.min(), y.max()
        if min_y:
            self.min_y = min_y
        if max_y:
            self.max_y = max_y
        self.labels = [(np.log(v + 1) - self.min_y) / (self.max_y - self.min_y) for v in self.labels]

        # self.labels = [torch.tensor([[target]]) for target in self.labels]

    def get_min_max_y(self):
        return self.min_y, self.max_y

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, idx):

        tree = self.trees[idx]
        count = self.labels[idx]
        inputs = self.inputs[idx]
        return tree, inputs, count
