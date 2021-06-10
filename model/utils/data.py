import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch


class Pollene1Dataset:
    def __init__(self, root: Path, mode: str, transform) -> None:
        self.transform = transform
        self.bboxes = pickle.load((root / f'annotations/new/{mode}.pkl').open('rb'))
        self.image_dir = root / mode
        self.images = list(sorted(self.bboxes.keys()))
        self.labels = ['pollen']
        # self.dims = np.array([640, 512, 640, 512])

    def __getitem__(self, idx):
        file = self.images[idx]
        img = cv2.imread(str(self.image_dir / file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self.bboxes[file]
        target = target.astype(float)
        # target[:, :4] /= self.dims

        im, bboxes, labels = self.transform(img, target[:, :4], target[:, 4])

        target = torch.hstack((bboxes, labels.unsqueeze(1)))
        return im, target

    def __len__(self):
        return len(self.images)


class HDF5Dataset:
    def __init__(self, root: Path, dataset: str, mode: str, transform) -> None:
        self.transform = transform
        self.data_file = root
        self.labels = ['poaceae', 'corylus', 'alnus']
        with h5py.File(self.data_file, 'r') as f:
            self.names = f[f'datasets/{dataset}/{mode}'][()]

    def _open_hdf5(self):
        self.dataset = h5py.File(self.data_file, 'r')
        self.images = self.dataset['images@2x']
        self.annotations = self.dataset['annotations@2x']

    def __getitem__(self, idx):
        if not hasattr(self, 'dataset'):
            self._open_hdf5()
        file = self.names[idx]
        img = self.images.get(file)[()]
        target = self.annotations.get(file)[()]

        target = target.astype(float)
        target = target[target[:, 4] != 3, :]
        im, bboxes, labels = self.transform(img, target[:, :4], target[:, 4])

        target = torch.hstack((bboxes, labels.unsqueeze(1)))
        return im, target

    def __len__(self):
        return self.names.size


class AstmaDataset:
    def __init__(self, root: Path, mode: str, transform) -> None:
        self.transform = transform
        self.bboxes = pickle.load((root / f'annotations/{mode}.pkl').open('rb'))
        self.image_dir = root / 'Images'
        self.images = list(sorted(self.bboxes.keys()))
        self.labels = ['poaceae', 'corylus', 'alnus']

    def __getitem__(self, idx):
        file = self.images[idx]
        img = cv2.imread(str(self.image_dir / file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (960, 540))

        target = self.bboxes[file]
        target = target.astype(float)
        target[:, :4] /= 2
        target = target[target[:, 4] != 3, ...]

        im, bboxes, labels = self.transform(img, target[:, :4], target[:, 4])

        target = torch.hstack((bboxes, labels.unsqueeze(1)))
        return im, target

    def __len__(self):
        return len(self.images)


class DataBatch:
    def __init__(self, data):
        images, target = list(zip(*data))
        self.images = torch.stack(images, dim=0)
        self.target = target

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.images = self.images.pin_memory()
        self.target = [t.pin_memory() for t in self.target]
        return self

    def data(self):
        return self.images, self.target


def collate(batch):
    return DataBatch(batch)
