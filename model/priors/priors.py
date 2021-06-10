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
