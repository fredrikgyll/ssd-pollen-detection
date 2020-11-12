import pickle
import torch
from pathlib import Path
from PIL import Image

from .augmentations import TransformerSequence


class Pollene1Dataset:
    def __init__(self, root: Path, transform: TransformerSequence) -> None:
        self.transform = transform
        self.bboxes = pickle.load((root / 'annotations/train_bboxes.pkl').open('rb'))
        image_dir = root / 'train'
        self.images = list(sorted(image_dir.glob('*.jpg')))

    def __getitem__(self, idx):
        file = self.images[idx]
        im = Image.open(file)
        bboxes = self.bboxes[file.name]
        labels = torch.ones(bboxes.size(0))
        im, bboxes, labels = self.transform(im, bboxes, labels)
        return im, bboxes, labels

    def __len__(self):
        return len(self.images)


def collate(batch):
    images, bboxes, labels = list(zip(*batch))
    return torch.stack(images, dim=0), bboxes, labels
