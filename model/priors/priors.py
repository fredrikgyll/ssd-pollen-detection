"""Calculates the default boxes (priors) for both SSD300 and SSD512"""
import torch

from typing import Dict, Any
from itertools import product
from math import sqrt
import numpy as np


def priors() -> torch.Tensor:
    """Generate default boxes for SSD300 or SSD512

    :return: Tensor of default boxes in center-size form
    :rtype: torch.Tensor

    Original:
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    # 5776 boxes layer 1 (5776)
    # 2166 boxes layer 2 (7942)
    #  600 boxes layer 3 (8542)
    #  150 boxes layer 4 (8692)
    #   36 boxes layer 5 (8728)
    #    4 boxes layer 6 (8732)
    """
    fig_size = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here:
    # https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [45, 70, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    fk = fig_size / np.array(steps)
    locations = []
    for idx, sfeat in enumerate(feat_size):
        sk1 = scales[idx] / fig_size
        sk2 = scales[idx + 1] / fig_size
        sk3 = sqrt(sk1 * sk2)
        all_sizes = [(sk1, sk1), (sk3, sk3)]

        for alpha in aspect_ratios[idx]:
            w, h = sk1 * sqrt(alpha), sk1 / sqrt(alpha)
            all_sizes.append((w, h))
            all_sizes.append((h, w))
        for w, h in all_sizes:
            for i, j in product(range(sfeat), repeat=2):
                cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                locations.append((cx, cy, w, h))
    dboxes = torch.tensor(locations, dtype=torch.float)
    return dboxes


class PriorBox:
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self):
        super(PriorBox, self).__init__()
        self.image_size = 300
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(self.aspect_ratios)
        self.variance = [0.1, 0.2]
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.steps = [8, 16, 32, 64, 100, 300]

        self.clip = True
        self.version = 'VOC'
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self) -> torch.Tensor:
        mean = []
        for k, f in enumerate(self.feature_maps):
            s_k = self.min_sizes[k] / self.image_size
            s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
            f_k = self.image_size / self.steps[k]

            boxes = [[s_k, s_k], [s_k_prime, s_k_prime]]
            for ar in self.aspect_ratios[k]:
                w, h = s_k * sqrt(ar), s_k / sqrt(ar)
                boxes.append([w, h])
                boxes.append([h, w])
            for default_w, default_h in boxes:

                for i, j in product(range(f), repeat=2):
                    # unit center x,y
                    cx = (j + 0.5) / f_k
                    cy = (i + 0.5) / f_k
                    mean.append([cx, cy, default_w, default_h])

        # back to torch land
        output = torch.tensor(mean)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

    def amdforward(self) -> torch.Tensor:
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


# 5776 boxes layer 1 (5776)
# 2166 boxes layer 2 (7942)
#  600 boxes layer 3 (8542)
#  150 boxes layer 4 (8692)
#   36 boxes layer 5 (8728)
#    4 boxes layer 6 (8732)
