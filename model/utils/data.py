import json
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from icecream import ic


class HDF5Dataset:
    def __init__(
        self, root: Path, dataset: str, mode: str, transform, sharpness_bound=None
    ) -> None:
        self.transform = transform
        self.data_file = root
        self.labels = ['poaceae', 'corylus', 'alnus']

        with h5py.File(self.data_file, 'r') as pollendb:
            self.names = list(pollendb[f'datasets/{dataset}/{mode}'][()])
            sharpness = pollendb['sharpness']
            lower, upper = sharpness_bound or [0, 1]
            self.targets = {}
            for file in self.names:
                target = pollendb['annotations@2x'].get(file)[()]
                target = target.astype(float)
                sharp = sharpness.get(file)[()]
                target_msk = (sharp > lower) & (sharp < upper)
                if target_msk.sum() == 0:
                    continue
                target = target[target_msk, ...]
                target_ids = np.flatnonzero(target_msk)
                labels = np.stack((target[:, 4], target_ids), 1)
                bboxes = target[:, :4]

                self.targets[file] = (bboxes, labels)
            new_names = set(self.targets.keys())
            pruned_names = set(self.names) - new_names
            if pruned_names:
                print(f'Pruned following images b/c of sharpness:\n{pruned_names}')
                self.names = list(new_names)

    def _open_hdf5(self):
        self.dataset = h5py.File(self.data_file, 'r')
        self.images = self.dataset['images@2x']

    def __getitem__(self, idx):
        if not hasattr(self, 'dataset'):
            self._open_hdf5()
        file = self.names[idx]
        img = self.images.get(file)[()]
        bbox, labels = self.targets[file]

        im, bboxes, labels = self.transform(img, bbox, labels)
        target = torch.hstack((bboxes, labels))
        return im, target

    def __len__(self):
        return len(self.names)


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
