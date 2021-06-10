import random

import numpy as np
import torch
import torchvision.transforms.functional as f


class TransformerSequence:
    def __init__(self, transformers) -> None:
        self.transformers = transformers

    def __call__(self, image, boxes, labels):
        for t in self.transformers:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels


class ToPercentCoords:
    def __call__(self, image, boxes, labels):
        assert image.size(0) == 3
        _, height, width = image.shape
        boxes[:, ::2] /= width
        boxes[:, 1::2] /= height

        return image, boxes, labels


class CV2Tensor:
    def __call__(self, image, boxes, labels):
        new_image = f.to_tensor(image)
        return new_image, boxes, labels


class TargetsToTensor:
    def __call__(self, image, boxes, labels):
        tensor_boxes = torch.FloatTensor(boxes)
        tensor_targets = torch.FloatTensor(labels)
        return image, tensor_boxes, tensor_targets


class Resize:
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes, labels):
        image = f.resize(image, (self.size, self.size))
        return image, boxes, labels


class Normalize:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __call__(self, image, boxes, labels):
        f.normalize(image, self.MEAN, std=self.STD, inplace=True)
        return image, boxes, labels


class DeNormalize:
    ALPHA = [-2.11790393, -2.03571429, -1.80444444]  # -mean/std
    BETA = [4.36681223, 4.46428571, 4.44444444]  # 1/std

    def __call__(self, image, boxes, labels):
        f.normalize(image, self.ALPHA, std=self.BETA, inplace=True)
        return image, boxes, labels


class RandomSubSample:
    def __init__(self, out_dim: int = 300, tolerance: int = 10) -> None:
        self.out_dim = out_dim
        self.tolerance = tolerance

    def __call__(self, image, boxes, labels):
        # print(image.shape)
        assert image.size(0) == 3
        dims = np.array(image.shape[1:])  # h, w
        out = self.out_dim

        if dims[0] == out and dims[1] == out:
            return image, boxes, labels
        deltas = dims - out

        while True:
            h, w = (np.random.rand(2) * deltas).astype(int)
            sub_img = image[..., h : h + out, w : w + out]
            new_boxes = boxes.copy()
            new_boxes[:, ::2] -= w
            new_boxes[:, 1::2] -= h
            in_bound = (new_boxes[:, :2] > -self.tolerance) & (
                new_boxes[:, 2:] < (self.out_dim + self.tolerance)
            )
            in_bound = in_bound.all(axis=1)
            if in_bound.any():
                break
        return sub_img, new_boxes[in_bound], labels[in_bound]


class StaticSubSample:
    def __init__(self, out_dim: int = 300, tolerance: int = 10) -> None:
        self.out_dim = out_dim
        self.tolerance = tolerance

    def __call__(self, image, boxes, labels):
        # print(image.shape)
        assert image.size(0) == 3
        dims = np.array(image.shape[1:])  # h, w
        out = self.out_dim

        if dims[0] == out and dims[1] == out:
            return image, boxes, labels
        deltas = dims - out
        h, w = 0, 0
        while True:
            sub_img = image[..., h : h + out, w : w + out]
            new_boxes = boxes.copy()
            new_boxes[:, ::2] -= w
            new_boxes[:, 1::2] -= h
            in_bound = (new_boxes[:, :2] > -self.tolerance) & (
                new_boxes[:, 2:] < (self.out_dim + self.tolerance)
            )
            in_bound = in_bound.all(axis=1)
            if in_bound.any():
                break
            w += deltas[1] // 5
            if w >= deltas[1]:
                w = 0
                h += deltas[0] // 5
                if h >= deltas[0]:
                    raise ValueError('Could not find valid SubSample')
        return sub_img, new_boxes[in_bound], labels[in_bound]


class HorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            _, _, width = image.shape
            boxes[:, 0::2] = width - boxes[:, 2::-2]
            image = f.hflip(image)
        return image, boxes, labels


class VerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            _, height, _ = image.shape
            boxes[:, 1::2] = height - boxes[:, 3::-2]
            image = f.vflip(image)
        return image, boxes, labels


class ChannelSuffle:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
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

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            random.shuffle(self.transforms)
            factors = np.random.uniform(0.5, 1.5, 3)
            for t, fac in zip(self.transforms, factors):
                image = t(image, fac)
            hue_shift = random.uniform(-0.3, 0.3)
            image = f.adjust_hue(image, hue_shift)

            return image, boxes, labels
        return image, boxes, labels


class SSDAugmentation(object):
    def __init__(self, size=300, train=True):
        self.size = size
        if train:
            transforms = [
                CV2Tensor(),  # boxes _ -> _
                ColorSift(),  # boxes _ -> _
                Normalize(),  # boxes _ -> _
                ChannelSuffle(),  # boxes _ -> _
                RandomSubSample(out_dim=500),  # boxes abs -> abs
                VerticalFlip(),  # boxes abs -> abs
                HorizontalFlip(),  # boxes abs -> abs
                ToPercentCoords(),  # boxes abs -> %
                Resize(size=self.size),  # boxes % -> %
                TargetsToTensor(),  # boxes _ -> _
            ]
        else:
            transforms = [
                CV2Tensor(),  # boxes _ -> _
                Normalize(),  # boxes _ -> _
                StaticSubSample(out_dim=500),  # boxes abs -> abs
                ToPercentCoords(),  # boxes abs -> %
                Resize(size=self.size),  # boxes % -> %
                TargetsToTensor(),  # boxes _ -> _
            ]
        self.augment = TransformerSequence(transforms)

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
