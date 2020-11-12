from typing import Any
import random

import torch
import torchvision.transforms.functional as f
import numpy as np


class TransformerSequence:
    def __init__(self, *transformers) -> None:
        self.transformers = transformers

    def __call__(self, image, boxes, labels):
        for t in self.transformers:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels


class ToStandardForm:
    def __call__(self, image, boxes, labels):
        new_boxes = torch.cat(
            (boxes[:, :2], boxes[:, 2:] + boxes[:, :2]), dim=1
        ).float()
        new_image = f.to_tensor(image)
        return new_image, new_boxes, labels


class Normalize:
    MEAN = [0.485, 0.456, 0.406]
    STD = std = [0.229, 0.224, 0.225]

    def __call__(self, image, boxes, labels):
        for i, (mean, std) in enumerate(zip(self.MEAN, self.STD)):
            image[i, ...] = (image[i, ...] - mean) / std
        return image, boxes, labels


class DeNormalize:
    MEAN = [0.485, 0.456, 0.406]
    STD = std = [0.229, 0.224, 0.225]

    def __call__(self, image, boxes, labels):
        for i, (mean, std) in enumerate(zip(self.MEAN, self.STD)):
            image[i, ...] = (image[i, ...] * std) + mean
        return image, boxes, labels


class SubSample:
    def __init__(self, width: int, height: int, out_dim: int = 300) -> None:
        self.out_dim = out_dim
        self.width = width
        self.height = height

    def __call__(self, image, boxes, labels) -> Any:
        # print(image.shape)
        dims = torch.tensor(image.shape[1:])  # h, w
        out = self.out_dim

        if dims[0] == out and dims[1] == out:
            return image, boxes, labels
        deltas = dims - out

        attempts = 0
        while True:
            attempts += 1
            h, w = (torch.rand(2) * deltas).int()
            sub_img = image[..., h : h + out, w : w + out]
            box_offset = torch.tensor([w, h, w, h])
            new_boxes = (boxes - box_offset) / float(self.out_dim)
            in_bound = (new_boxes[:, :2] > -0.05) & (new_boxes[:, 2:] < 1.05)
            in_bound = in_bound.all(dim=1)
            if in_bound.any().item():
                break
        # print(attempts)
        return sub_img, new_boxes[in_bound], labels[in_bound]


class HorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, boxes, labels) -> Any:
        random_val = torch.rand(1).item()
        if random_val < self.p:
            new_boxes = torch.stack(
                (1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]), dim=1
            )
            new_image = f.hflip(image)
            return new_image, new_boxes, labels
        return image, boxes, labels


class VerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, boxes, labels) -> Any:
        random_val = torch.rand(1).item()
        if random_val < self.p:
            new_boxes = torch.stack(
                (boxes[:, 0], 1 - boxes[:, 3], boxes[:, 2], 1 - boxes[:, 1]), dim=1
            )
            new_image = f.vflip(image)
            return new_image, new_boxes, labels
        return image, boxes, labels


class ChannelSuffle:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, boxes, labels) -> Any:
        random_val = torch.rand(1).item()
        if random_val < self.p:
            channel_order = torch.arange(0, image.size(0))
            random.shuffle(channel_order)
            return image[channel_order, ...], boxes, labels
        return image, boxes, labels


class ColorSift:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
        self.transforms = [
            f.adjust_brightness,
            f.adjust_contrast,
            f.adjust_saturation,
        ]

    def __call__(self, image, boxes, labels) -> Any:
        random_val = torch.rand(1).item()
        if random_val < self.p:
            random.shuffle(self.transforms)
            factors = np.random.uniform(0.5, 1.5, 3)
            for t, fac in zip(self.transforms, factors):
                image = t(image, fac)
            return image, boxes, labels
        return image, boxes, labels


def get_transform(train: bool = True):
    transforms = [
        ToStandardForm(),
        SubSample(640, 512),
        Normalize(),
    ]
    if train:
        transforms += [
            VerticalFlip(),
            HorizontalFlip(),
            ColorSift(),
            ChannelSuffle(),
        ]
    return TransformerSequence(*transforms)
