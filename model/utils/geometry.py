"""Helper functions for bounding box calculations"""
from typing import Tuple, Dict, Any, Union, List

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import softmax
from torch.nn.modules.activation import Threshold


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
    boxes_out = torch.zeros(defaults.size()).float()
    pos_mask = matched_targets.ge(0)
    # print(f'positive boxes: {pos_mask.sum()}')
    labels_out[pos_mask] = labels[matched_targets[pos_mask]].int()
    boxes_out[pos_mask] = center_size(truths[matched_targets[pos_mask]])
    return boxes_out.transpose(0, 1), labels_out


def rescale_batch(
    defaults: Tensor, ploc: Tensor, pconf: Tensor
) -> Tuple[Tensor, Tensor]:
    ploc = ploc.transpose(1, 2)  # (N, nboxes, 4)
    pconf = pconf.transpose(1, 2)  # (N, nboxes, nclasses)

    # ploc has offsets relative to the default boxcoordinates so must be transformed
    # to actual bounding boxes
    dboxes = defaults.unsqueeze(dim=0)

    ploc[:, :, :2] = 0.1 * ploc[:, :, :2]
    ploc[:, :, 2:] = 0.2 * ploc[:, :, 2:]

    ploc[:, :, :2] = ploc[:, :, :2] * dboxes[:, :, 2:] + dboxes[:, :, :2]
    ploc[:, :, 2:] = ploc[:, :, 2:].exp() * dboxes[:, :, 2:]

    ploc = torch.cat(
        (
            ploc[:, :, :2] - (ploc[:, :, 2:] / 2.0),
            ploc[:, :, :2] + (ploc[:, :, 2:] / 2.0),
        ),
        dim=2,
    )

    return ploc, F.softmax(pconf, dim=2)

def nms(
    bbox: Tensor,
    prob: Tensor,
    soft: bool,
    iou_thr: float,
    score_thr: float,
    max_num: int,
) -> Tensor:
    out_boxes = []
    out_labels = []
    out_confs = []
    for i, label_prob in enumerate(prob.split(1, 1)[1:]):
        scores = label_prob.squeeze(1)
        scores_sorted, scores_sorted_idx = scores.sort(dim=0, descending=True)
        # return bbox[scores_sorted_idx[:100]], 0, 0 # Get all top boxes
        picks = []
        while scores_sorted_idx.numel():
            chosen_idx = scores_sorted_idx[0]
            chosen = bbox[chosen_idx].unsqueeze(0)
            picks.append(chosen_idx)
            iou = jaccard(chosen, bbox[scores_sorted_idx]).squeeze()
            iou_mask = iou > iou_thr
            if soft:
                scores_sorted[iou_mask] *= 1-scores_sorted[iou_mask]
                score_mask = scores_sorted > score_thr
                scores_sorted_idx = scores_sorted_idx[score_mask]
                scores_sorted = scores_sorted[score_mask]
                _, new_order = scores_sorted.sort(dim=0, descending=True)
                scores_sorted_idx = scores_sorted_idx[new_order]
            else:
                scores_sorted_idx = scores_sorted_idx[~iou_mask]


        out_boxes.append(bbox[picks, :])
        out_confs.append(scores[picks])
        out_labels.extend([i + 1] * len(picks))

    if not out_boxes:
        return (torch.tensor([]) for _ in range(3))

    out_boxes, out_confs, out_labels = (
        torch.cat(out_boxes, 0),
        torch.cat(out_confs, 0),
        torch.tensor(out_labels, dtype=torch.long),
    )

    _, indexes = out_confs.sort(descending=True)
    indexes = indexes[:max_num]
    return (out_boxes[indexes, :], out_confs[indexes], out_labels[indexes])


def decode(
    defaults: Tensor,
    ploc: Tensor,
    pconf: Tensor,
    soft: bool = True,
    iou_thr: float = 0.5,
    score_thr: float = 0.001,
    max_num: int = 25,
) -> Tensor:
    """Transform the model output into bounding boxes and perform NMS on each image"""

    bboxes, probs = rescale_batch(defaults, ploc, pconf)

    output_boxes = []
    output_labels = []
    output_cofs = []
    for bbox, prob in zip(bboxes.split(1, dim=0), probs.split(1, dim=0)):
        bbox = bbox.squeeze(0)
        prob = prob.squeeze(0)
        boxes, conf, labels = nms(bbox, prob, True, 0.2, 0.05, 10)
        # boxes, conf, labels = decode_single(bbox, prob, criteria=iou_thr, max_output=max_num, max_num=50)

        output_boxes.append(boxes)
        output_cofs.append(conf)
        output_labels.append(labels)
    return (output_boxes, output_cofs, output_labels)
