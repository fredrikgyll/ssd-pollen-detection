"""Helper functions for bounding box calculations"""
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from icecream import ic


def point_form(boxes: Tensor):
    """Convert boxes to (xmin, ymin, xmax, ymax) representation.

    :param boxes: Center-size default boxes from priorbox layers.
    :type boxes: class:`torch.Tensor`
    :return: Converted xmin, ymin, xmax, ymax form of boxes.
    :rtype: class:`torch.Tensor`
    """
    # assert boxes.size(1) == 4

    return torch.cat(
        (
            boxes[:, :2] - boxes[:, 2:] / 2.0,
            boxes[:, :2] + boxes[:, 2:] / 2.0,
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
    # assert boxes.size(1) == 4

    return torch.cat(
        (
            (boxes[:, 2:] + boxes[:, :2]) / 2.0,
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
    # assert boxes_a.size(1) == 4
    # assert boxes_b.size(1) == 4

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


def area(boxes: Tensor) -> Tensor:
    """Return Tensor of areas of each box

    :param boxes: Tensor of boxes in point form,
        i.e. [(xmin, ymin, xmax, ymax),...]
    :type boxes: class:`torch.Tensor`
    :return: Tensor of area values in [0,1]
    :rtype: Tensor[boxes.size]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def jaccard(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Return jaccard overlap (IoU) of boxes in boxes_a and boxes_b

    :param boxes_a: Tensor of boxes in point form,
        i.e. [(xmin, ymin, xmax, ymax),...]
    :type boxes_a: class:`torch.Tensor`
    :param boxes_b: Tensor of boxes in point form,
        i.e. [(xmin, ymin, xmax, ymax),...]
    :type boxes_b: class:`torch.Tensor`
    :return: Tensor of IoU values where element (i,j)
        is intersecion[0,1] of boxes_a[i] and boxes_b[j]
    :rtype: Tensor[boxes_a.size, boxes_b.size]
    """
    # assert boxes_a.size(1) == 4
    # assert boxes_b.size(1) == 4

    inter = intersect(boxes_a, boxes_b)
    # Area is (x_max - x_min) * (y_max - y_min)
    # unsqueze so we can combine them with intersection.
    # boxes_a is expanded so each row is a copy of the areas
    # boxes_b is expanded so every column is a copy of the areas
    areas_a = area(boxes_a).unsqueeze(1).expand_as(inter)  # [A,B]
    areas_b = area(boxes_b).unsqueeze(0).expand_as(inter)  # [A,B]

    union = areas_a + areas_b - inter
    return inter / union  # [A,B]


def encode(
    defaults: Tensor, truths: Tensor, labels: Tensor, threshhold: float = 0.5
) -> Tensor:
    """Return encoded form of target bboxes and labels mached with default boxes

    :param defaults: Default bboxes on center form,
        i.e. [(cx, cy, w, h), ...]
    :type defaults: class:`torch.Tensor`
    :param truths: Ground truth boxes in point form
        i.e. [(xmin, ymin, xmax, ymax),...]
    :type truths: class:`torch.Tensor`
    :param labels: Target labels for the corresponding gt boxes
    :type labels: class:`torch.Tensor`
    :param threshhold: Minimum IoU to match default box to any gt box, defaults to 0.5
    :type threshhold: float, optional
    :return: Tuple of (bboxes, labels) for loss calculations
    :rtype: Tuple[Tensor, Tensor]
    """
    # assert defaults.size() == torch.Size([8732, 4])
    # assert truths.size(0) == labels.size(0)

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
    boxes_out = defaults.clone()
    pos_mask = matched_targets.ge(0)
    # print(f'positive boxes: {pos_mask.sum()}')
    labels_out[pos_mask] = labels[matched_targets[pos_mask]].int() + 1
    boxes_out[pos_mask] = center_size(truths[matched_targets[pos_mask]])
    return boxes_out.transpose(0, 1), labels_out


def _gausian_penelty(iou, scores, sigma=0.2, **kwargs):
    exp = iou ** 2 / sigma
    scores *= torch.exp(-exp)
    return scores.sort(descending=True)


def _linear_penelty(iou, scores, threshold=0.5, **kwargs):
    iou_mask = iou.ge(threshold)
    scores[iou_mask] *= 1 - iou[iou_mask]
    return scores.sort(descending=True)


def _original_penelty(iou, scores, threshold=0.5, **kwargs):
    iou_mask = iou.le(threshold)
    return scores, iou_mask


def nms(
    boxes: Tensor, scores: Tensor, overlap=0.5, sigma=0.1, top_k=200, penelty='gausian'
):
    """Apply Non-maximum Suppression to boxes based on score value
    :param boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
    :param scores: (tensor) The class predscores for the img, Shape:[num_priors].
    :param overlap: (float) The overlap thresh for suppressing unnecessary boxes.
    :param top_k: (int) The Maximum number of box preds to consider.
    :returns: The indices of the kept boxes with respect to num_priors.
    """
    keep = []
    rescore = {
        'gausian': _gausian_penelty,
        'linear': _linear_penelty,
        'original': _original_penelty,
    }[penelty]

    sorted_scores, sorted_scores_idx = scores.sort(descending=True)
    sorted_boxes = boxes[sorted_scores_idx]

    while sorted_boxes.numel() and (sorted_scores.ge(0.05)).any():
        keep.append(sorted_scores_idx[0].item())
        selected = boxes[keep[-1]]
        iou = jaccard(selected.unsqueeze(0), sorted_boxes).squeeze()

        sorted_scores, rescoring_idx = rescore(
            iou, sorted_scores, threshold=overlap, sigma=sigma
        )

        sorted_scores_idx = sorted_scores_idx[rescoring_idx]
        sorted_boxes = sorted_boxes[rescoring_idx]

    keep = torch.tensor(keep)[:top_k]
    count = len(keep)
    return keep, count
