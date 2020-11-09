"""Helper functions for bounding box calculations"""
from typing import Tuple, Dict, Any, Union, List

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.activation import Threshold


def point_form(boxes: Tensor):
    """Convert boxes to (xmin, ymin, xmax, ymax) representation.

    :param boxes: Center-size default boxes from priorbox layers.
    :type boxes: class:`torch.Tensor`
    :return: Converted xmin, ymin, xmax, ymax form of boxes.
    :rtype: class:`torch.Tensor`
    """
    assert boxes.size(1) == 4

    return torch.cat(
        (
            boxes[:, :2] - boxes[:, 2:] / 2,
            boxes[:, :2] + boxes[:, 2:] / 2,
        ),
        dim=1,
    )


def center_size(boxes: Tensor) -> Tensor:
    """Convert boxes to center-size (cx, cy, w, h) representation.

    :param boxes: Point form (xmin, ymin, xmax, ymax) boxes
    :type boxes: class:`torch.Tensor`
    :return: Converted (cx, cy, w, h) form of boxes.
    :rtype: class:`torch.Tensor`
    """
    assert boxes.size(1) == 4

    return torch.cat(
        (
            (boxes[:, 2:] + boxes[:, :2]) / 2,
            boxes[:, 2:] - boxes[:, :2],
        ),
        dim=1,
    )


def intersect(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Return intersection of boxes in boxes_a and boxes_b

    :param boxes_a: Tensor of boxes in point form,
        i.e. [(xmin, ymin, xmax, ymax),...]
    :type boxes_a: class:`torch.Tensor`
    :param boxes_b: Tensor of boxes in point form,
        i.e. [(xmin, ymin, xmax, ymax),...]
    :type boxes_b: class:`torch.Tensor`
    :return: Tensor of intersect values where element (i,j)
        is intersecion[0,1] of boxes_a[i] and boxes_b[j]
    :rtype: Tensor[boxes_a.size, boxes_b.size]
    """
    assert boxes_a.size(1) == 4
    assert boxes_b.size(1) == 4

    A = boxes_a.size(0)
    B = boxes_b.size(0)
    max_xy = torch.min(
        boxes_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        boxes_b[:, 2:].unsqueeze(0).expand(A, B, 2),
    )
    min_xy = torch.max(
        boxes_a[:, :2].unsqueeze(1).expand(A, B, 2),
        boxes_b[:, :2].unsqueeze(0).expand(A, B, 2),
    )
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Return Tensor of floats with the jaccard overlap
    (IoU) of boxes_a and boxes_b

    return:
        Tensor[boxes_a.size, boxes_b.size] where
        element (i,j) is IoU of boxes_a[i] and boxes_b[j]
    """
    assert boxes_a.size(1) == 4
    assert boxes_b.size(1) == 4

    inter = intersect(boxes_a, boxes_b)
    # Area is (x_max - x_min) * (y_max - y_min)
    # unsqueze so we can combine them with intersection.
    # boxes_a is expanded so each row is a copy of the areas
    # boxes_b is expanded so every column is a copy of the areas
    areas_a = (
        ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))
        .unsqueeze(1)
        .expand_as(inter)
    )  # [A,B]
    areas_b = (
        ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))
        .unsqueeze(0)
        .expand_as(inter)
    )  # [A,B]
    union = areas_a + areas_b - inter
    return inter / union  # [A,B]


def match(
    defaults: Tensor, truths: Tensor, labels: Tensor, threshhold: float = 0.5
) -> Tensor:
    assert defaults.size() == torch.Size([8732, 4])
    assert truths.size(0) == labels.size(0)

    overlap = jaccard(point_form(defaults), truths)
    # best_default, best_defailt_idx = torch.max(overlap, 0, keepdim=True)
    # matches = overlap.ge(best_default) | overlap.ge(threshhold)
    # return matches
    #  element i in matched_targets is idx of matched truth box
    matched_targets = torch.full((defaults.size(0),), -1)

    # highest jaccard overlap for every default box
    best_target, best_target_idx = overlap.max(1)
    best_default, best_default_idx = overlap.max(0)

    defaults_mask = best_target > threshhold

    matched_targets[defaults_mask] = best_target_idx[defaults_mask]
    matched_targets.index_copy_(0, best_default_idx, torch.arange(0, truths.size(0)))

    labels_out = torch.zeros(defaults.size(0)).int()
    boxes_out = torch.zeros(defaults.size()).float()
    pos_mask = matched_targets.ge(0)
    labels_out[pos_mask] = labels[matched_targets[pos_mask]].int()
    boxes_out[pos_mask] = center_size(truths[matched_targets[pos_mask]])
    return boxes_out, labels_out
