import numpy as np
import torch
from math import ceil


class TensorDataLoader:
    """Combination of torch's DataLoader and TensorDataset for efficient batch
    sampling and adaptive augmentation on GPU.
    """

    def __init__(self, x, y, transform=None, transform_y=None, batch_size=500,
                 data_factor=1, shuffle=False, detach=False):
        assert x.size(0) == y.size(0), 'Size mismatch'
        self.x = x
        self.y = y
        self.device = x.device
        self.data_factor = data_factor
        self.n_data = y.size(0)
        self.batch_size = batch_size
        self.n_batches = ceil(self.n_data / self.batch_size)
        self.shuffle = shuffle
        identity = lambda x: x
        self.transform = transform if transform is not None else identity
        self.transform_y = transform_y if transform_y is not None else identity
        self._detach = detach

    def __iter__(self):
        if self.shuffle:
            permutation = torch.randperm(self.n_data, device=self.device)
            self.x = self.x[permutation]
            self.y = self.y[permutation]
        self.i_batch = 0
        return self

    def __next__(self):
        if self.i_batch >= self.n_batches:
            raise StopIteration

        start = self.i_batch * self.batch_size
        end = start + self.batch_size
        if self._detach:
            x = self.transform(self.x[start:end]).detach()
        else:
            x = self.transform(self.x[start:end])
        y = self.transform_y(self.y[start:end])
        self.i_batch += 1
        return (x, y)

    def __len__(self):
        return self.n_batches

    def attach(self):
        self._detach = False
        return self

    def detach(self):
        self._detach = True
        return self

    @property
    def dataset(self):
        return DatasetDummy(self.n_data * self.data_factor)


class SubsetTensorDataLoader(TensorDataLoader):

    def __init__(self, x, y, transform=None, transform_y=None, subset_size=500,
                 data_factor=1, detach=False):
        self.subset_size = subset_size
        super().__init__(x, y, transform, transform_y, batch_size=subset_size,
                         data_factor=data_factor, shuffle=True, detach=detach)
        self.n_batches = 1  # -> len(loader) = 1

    def __iter__(self):
        self.i_batch = 0
        return self

    def __next__(self):
        if self.i_batch >= self.n_batches:
            raise StopIteration

        sod_indices = np.random.choice(self.n_data, self.subset_size, replace=False)
        if self._detach:
            x = self.transform(self.x[sod_indices]).detach()
        else:
            x = self.transform(self.x[sod_indices])
        y = self.transform_y(self.y[sod_indices])
        self.i_batch += 1
        return (x, y)

    def __len__(self):
        return 1

    @property
    def dataset(self):
        return DatasetDummy(self.subset_size * self.data_factor)

        
class GroupedSubsetTensorDataLoader(SubsetTensorDataLoader):
    
    def __init__(self, x, y, transform=None, transform_y=None, subset_size=500, data_factor=1, detach=False):
        super().__init__(x, y, transform, transform_y, subset_size, data_factor, detach)
        self.class_indices = list()
        for c in torch.unique(self.y):
            self.class_indices.append((self.y == c).nonzero().squeeze())
        self.n_classes = len(self.class_indices)

    def __next__(self):
        if self.i_batch >= self.n_batches:
            raise StopIteration

        # class order
        # NOTE: assumes roughly stratified classes
        class_order = np.random.permutation(self.n_classes)
        sod_indices = list()
        n_data_remaining = self.subset_size
        for c in class_order:
            if n_data_remaining == 0:
                break
            local_indices = self.class_indices[c]
            local_shuffle = torch.randperm(len(local_indices))
            c_indices = local_indices[local_shuffle][:n_data_remaining]
            n_data_remaining -= len(c_indices)
            sod_indices.append(c_indices)
        sod_indices = torch.cat(sod_indices)

        if self._detach:
            x = self.transform(self.x[sod_indices]).detach()
        else:
            x = self.transform(self.x[sod_indices])
        y = self.transform_y(self.y[sod_indices])
        self.i_batch += 1
        return (x, y)


class DatasetDummy:
    def __init__(self, N):
        self.N = N

    def __len__(self):
        return int(self.N)


def dataset_to_tensors(dataset, indices=None, device='cuda'):
    if indices is None:
        indices = range(len(dataset))  # all
    xy_train = [dataset[i] for i in indices]
    x = torch.stack([e[0] for e in xy_train]).to(device)
    y = torch.stack([torch.tensor(e[1]) for e in xy_train]).to(device)
    return x, y
