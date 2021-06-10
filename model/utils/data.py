import pickle
import torch
from pathlib import Path
import cv2
import numpy as np


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


class AstmaDataset:
    def __init__(self, root: Path, mode: str, transform) -> None:
        self.transform = transform
        self.bboxes = pickle.load((root / f'annotations/{mode}.pkl').open('rb'))
        self.image_dir = root / 'Images'
        self.images = list(sorted(self.bboxes.keys()))
        self.labels = ['poaceae', 'corylus', 'alnus', 'unknown']

    def __getitem__(self, idx):
        file = self.images[idx]
        img = cv2.imread(str(self.image_dir / file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self.bboxes[file]
        target = target.astype(float)

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
