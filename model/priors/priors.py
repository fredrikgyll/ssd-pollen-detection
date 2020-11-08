"""Calculates the default boxes (priors) for both SSD300 and SSD512"""
import torch

from typing import Dict, Any
from itertools import product
from math import sqrt


def priors(cfg: Dict[str, Any]) -> torch.Tensor:
    """Generate default boxes for SSD300 or SSD512

    :param cfg: configuration dictionary
    :type cfg: Dict[str, Any]
    :return: Tensor of default boxes in center-size form
    :rtype: torch.Tensor
    """
    feature_maps = cfg['feature_maps']
    locations = []
    s_min = cfg['s_min']
    s_max = cfg['s_max']
    scale_factor = (s_max - s_min) / (len(feature_maps) - 1)
    ratios = [[sqrt(a_r) for a_r in a] for a in cfg['aspect_ratios']]
    s_k = 0
    s_k_1 = s_min
    for k, (f, a) in enumerate(zip(cfg['feature_maps'], ratios)):
        s_k = s_k_1
        s_k_1 += scale_factor
        s_k_prime = sqrt(s_k * s_k_1)
        for i, j in product(range(f), repeat=2):
            cy = (i + 0.5) / f
            cx = (j + 0.5) / f

            locations.append([cx, cy, s_k, s_k])
            locations.append([cx, cy, s_k_prime, s_k_prime])

            for a_r in a:
                locations.append([cx, cy, s_k * a_r, s_k / a_r])  # case: a_r = 2
                locations.append([cx, cy, s_k / a_r, s_k * a_r])  # case: a_r = 1/2
    return torch.tensor(locations)
