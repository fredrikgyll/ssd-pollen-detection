import json
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from icecream import ic


class HDF5Dataset:
    def __init__(self, root: Path, dataset: str, mode: str, transform, sharpness_bound = None) -> None:
        self.transform = transform
        self.data_file = root
        self.labels = ['poaceae', 'corylus', 'alnus']
        self.sharpness_bound = sharpness_bound or [0, 1]
        with h5py.File(self.data_file, 'r') as f:
            self.names = f[f'datasets/{dataset}/{mode}'][()]

    def _open_hdf5(self):
        self.dataset = h5py.File(self.data_file, 'r')
        self.images = self.dataset['images@2x']
        self.annotations = self.dataset['annotations@2x']
        self.sharpness = self.dataset['sharpness']

    def __getitem__(self, idx):
        if not hasattr(self, 'dataset'):
            self._open_hdf5()
        file = self.names[idx]
        img = self.images.get(file)[()]
        target = self.annotations.get(file)[()]
        if self.sharpness_bound:
            target_msk = (target > self.sharpness_bound[0]) & (target < self.sharpness_bound[1])
            target = self.annotations.get(file)[target_msk]
            target_ids = np.flatnonzero(target_msk)
        else:
            target_ids = np.arange(target.shape[0])

        target = target.astype(float)
        labels = np.stack((target[:, 4], target_ids), 1)
        im, bboxes, labels = self.transform(img, target[:, :4], labels)
        target = torch.hstack((bboxes, labels))
        return im, target

    def __len__(self):
        return self.names.size


class AstmaDataset:
    def __init__(self, root: Path, name: str, mode: str, transform) -> None:
        self.transform = transform
        self.bboxes = pickle.loads((root / 'annotations/all.pkl').read_bytes())
        self.images = json.loads((root / f'datasets/{name}.json').read_text())[mode]
        self.image_dir = root / 'Images'
        self.labels = ['poaceae', 'corylus', 'alnus']

    def __getitem__(self, idx):
        file = self.images[idx]
        img = cv2.imread(str(self.image_dir / file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (960, 540))

        target = self.bboxes[file]
        target = target.astype(float)
        target[:, :4] /= 2

        labels = np.stack((target[:, 4], np.arange(target.shape[0])), 1)

        im, bboxes, labels = self.transform(img, target[:, :4], labels)

        target = torch.hstack((bboxes, labels))
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
