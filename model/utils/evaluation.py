from collections import defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm
from icecream import ic

from model.utils.geometry import jaccard


def interpolate(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    """Return list of 11-point interpolation points

    :param precision: precision by rising recall
    :type precision: class:`np.ndarray`
    :param recall: monotonically increasing recall levels
    :type recall: class:`np.ndarray`
    :returns: list of 11-point interpolation
    :rtype: class:`np.ndarray`
    """
    pre = np.append(precision, 0.0)
    rec = np.append(recall, 1.0)
    interpolation = []
    for p in np.linspace(0, 1, 11):
        interpolation.append(np.max(pre[rec >= p]))
    return np.array(interpolation)


def calculate_true_positives(
    detections: torch.Tensor, ground_truths: torch.Tensor
) -> torch.Tensor:
    """Return list labeling detections as TP

    :param detections: Detections in point form [N,4], ordered by precedence
        i.e. competing detections are resolved by lowest index
    :type detections: class:`torch.Tensor`
    :param ground_truths: ground truths in point form [M, 4].
    :type detections: class:`torch.Tensor`
    :return: Tensor[N] containing 1 if the corresponding detection is a TP
    :rtype: `torch.Tensor`
    """
    iou = jaccard(detections, ground_truths)
    tps = torch.zeros(detections.size(0), dtype=int)
    for ground_column in iou.split(1, 1):
        tp = torch.nonzero(ground_column.squeeze(-1) > 0.5).squeeze(-1)
        if tp.nelement() > 0:
            tps[tp[0]] = 1
    return tps


def evaluate(model, dataset, class_subset, quiet=True):
    length = len(dataset)

    n_gth = defaultdict(int)
    predictions = defaultdict(list)
    confidences = defaultdict(list)

    model.eval()
    model = model.cuda()

    for i in tqdm(range(length), disable=quiet):
        try:
            image, targets = dataset[i]
        except ValueError:
            continue
        image = image.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            detections = model(image.unsqueeze(0))
        for j, name in enumerate(dataset.labels):
            if name not in class_subset:
                continue  # skip classes
            truths = targets[targets[:, 4] == j, :4]
            n_gth[name] += len(truths)
            dets = detections[0, j + 1, ...]  # 0 is bkg_label
            mask = dets[:, 0].gt(0.0)
            if mask.any():
                dets = dets[mask, ...]
                sorted_conf, order_idx = dets[:, 0].sort(descending=True)
                tps = calculate_true_positives(dets[order_idx, 1:], truths)
                predictions[name].append(tps)
                confidences[name].append(sorted_conf)

    metrics = {}
    for name in class_subset:
        true_pos = torch.cat(predictions[name], dim=0)
        confs = torch.cat(confidences[name], dim=0)
        _, order = confs.sort(descending=True)
        true_pos_cum = torch.cumsum(true_pos[order], dim=0).cpu().numpy()

        precision = true_pos_cum / np.arange(1, true_pos_cum.size + 1)
        recall = true_pos_cum / n_gth[name]
        interpolation = interpolate(precision, recall)
        metrics[name] = {
            'precision': precision,
            'recall': recall,
            'interpolation': interpolation,
            'average_precision': np.mean(interpolation),
            'ground_truths': n_gth[name],
            'total_detections': true_pos.size(0),
            'tp': true_pos.sum().item(),
            'fp': (true_pos == 0).sum().item(),
        }

    return metrics
