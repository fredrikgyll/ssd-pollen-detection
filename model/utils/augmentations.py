from typing import Any
from matplotlib.pyplot import box

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
        new_boxes = np.hstack((boxes[:, :2], boxes[:, 2:] + boxes[:, :2]))
        return image.transpose(2, 0, 1), new_boxes, labels


class FromIntToFloat:
    def __call__(self, image, boxes, labels):
        return image.astype(np.float32), boxes.astype(np.float32), labels


class SubtractMean:
    VGG_MEAN = [123.68, 116.78, 103.94]

    def __init__(self, means=[123.68, 116.78, 103.94]) -> None:
        self.means = np.array(means)

    def __call__(self, image, boxes, labels):
        new_image = image - self.means
        # print(new_image[:5, :5, :])
        return new_image, boxes, labels


class SubSample:
    def __init__(self, width: int, height: int, out_dim: int = 300) -> None:
        self.out_dim = out_dim
        self.width = width
        self.height = height

    def __call__(self, image, boxes, labels) -> Any:
        # print(image.shape)
        _, *dims = image.shape  # h, w
        out = self.out_dim
        dims = np.array(dims)

        if dims[0] == out and dims[1] == out:
            return image, boxes, labels
        deltas = dims - out

        attempts = 0
        while True:
            attempts += 1
            h, w = (np.random.rand(2) * deltas).astype(int)
            sub_img = image[..., h : h + out, w : w + out]
            new_boxes = (boxes - [w, h, w, h]) / float(self.out_dim)
            in_bound = (new_boxes[:, :2] > -0.1) & (new_boxes[:, 2:] < 1.1)
            in_bound = in_bound.all(axis=1)
            if in_bound.any():
                break
        # print(attempts)
        return sub_img, new_boxes[in_bound], labels[in_bound]


class HorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, boxes, labels) -> Any:
        random_val = np.random.random()
        if random_val < self.p:
            new_boxes = np.stack(
                (1 - boxes[:, 0], boxes[:, 1], 1 - boxes[:, 2], boxes[:, 3]), axis=1
            )
            new_image = image[..., ::-1]
            return new_image, new_boxes, labels
        return image, boxes, labels


class ChannelSuffle:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, boxes, labels) -> Any:
        random_val = np.random.random()
        if random_val < self.p:
            channel_order = np.arange(0, image.shape[0])
            np.random.shuffle(channel_order)
            return image[channel_order, ...], boxes, labels
        return image, boxes, labels
