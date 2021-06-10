import pickle
import torch
from pathlib import Path
import cv2
from .augmentations import TransformerSequence


class APollene1Dataset:
    def __init__(self, root: Path, mode: str, transform: TransformerSequence) -> None:
        self.transform = transform
        self.bboxes = pickle.load((root / f'annotations/{mode}.pkl').open('rb'))
        self.image_dir = root / mode
        self.images = list(sorted(self.bboxes.keys()))

    def __getitem__(self, idx):
        file = self.images[idx]
        im = cv2.imread(str(self.image_dir / file))
        bboxes = self.bboxes[file]
        labels = torch.ones(bboxes.size(0))
        im, bboxes, labels = self.transform(im, bboxes, labels)
        return im, bboxes, labels

    def __len__(self):
        return len(self.images)


class Pollene1Dataset:
    def __init__(self, root: Path, mode: str, transform: TransformerSequence) -> None:
        self.transform = transform
        self.bboxes = pickle.load((root / f'annotations/{mode}.pkl').open('rb'))
        self.image_dir = root / mode
        self.images = list(sorted(self.bboxes.keys()))

    def __getitem__(self, idx):
        file = self.images[idx]
        im = cv2.imread(str(self.image_dir / file))
        bboxes = self.bboxes[file]
        labels = torch.ones(bboxes.size(0))
        im, bboxes, labels = self.transform(im, bboxes, labels)
        return im, bboxes, labels

    def __len__(self):
        return len(self.images)


class DataBatch:
    def __init__(self, data):
        images, bboxes, labels = list(zip(*data))
        self.images = torch.stack(images, dim=0)
        self.bboxes = bboxes
        self.labels = labels

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.images = self.images.pin_memory()
        self.bboxes = [box.pin_memory() for box in self.bboxes]
        self.labels = [label.pin_memory() for label in self.labels]
        return self

    def data(self):
        return self.images, self.bboxes, self.labels


def collate(batch):
    return DataBatch(batch)
