"""Calculates the default boxes (priors) for both SSD300 and SSD512"""
from itertools import product
from math import sqrt

import numpy as np
import torch


class PriorBox:
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        self.image_size = cfg['size']
        self.layer_activation = cfg['layer_activation']
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        # number of priors for feature map location (either 4 or 6)
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.min_sizes = [30, 40, 60, 100, 150, 200]
        self.max_sizes = [40, 60, 100, 150, 200, 264]
        self.steps = [8, 16, 32, 64, 100, 300]

        self.clip = True

    def forward(self) -> torch.Tensor:
        mean = []
        for k, f in enumerate(self.feature_maps):
            if not self.layer_activation[k]:
                continue
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

        output = torch.tensor(mean).transpose(0, 1)  # [4, -1]
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


# 5776 boxes layer 1 (5776)
# 2166 boxes layer 2 (7942)
#  600 boxes layer 3 (8542)
#  150 boxes layer 4 (8692)
#   36 boxes layer 5 (8728)
#    4 boxes layer 6 (8732)
