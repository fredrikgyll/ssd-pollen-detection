from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm
from icecream import ic

from model.utils.augmentations import UnAugment
from model.utils.geometry import jaccard
from model.utils.sharpness import image_bbox_quality


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

def calculate_positive_localizations(
    detections: torch.Tensor, ground_truths: torch.Tensor
) -> torch.Tensor:
    """Return list labeling detections as positive localizations
    i.e. whether or not the detection matches a GT 

    :param detections: Detections in point form [N,4], ordered by precedence
        i.e. competing detections are resolved by lowest index
    :type detections: class:`torch.Tensor`
    :param ground_truths: ground truths in point form [M, 4].
    :type detections: class:`torch.Tensor`
    :return: Tensor[N] containing True if the corresponding detection matches a GT
    :rtype: `torch.Tensor`
    """
    iou = jaccard(detections, ground_truths)
    positive = torch.any(iou > 0.5, dim=1)
    return positive

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
    tps = torch.full((detections.size(0),), -1, dtype=int)
    for i, ground_column in enumerate(iou.split(1, 1)):
        tp = torch.nonzero(ground_column.squeeze(-1) > 0.5).squeeze(-1)
        if tp.nelement() > 0:
            tps[tp[0]] = i
    return tps


def calculate_detection_sharpness(detections: torch.Tensor, image: np.ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    bbox = (
        torch.stack(
            (
                detections[:, 0] * w,
                detections[:, 1] * h,
                detections[:, 2] * w,
                detections[:, 3] * h,
            ),
            dim=1,
        )
        .int()
        .cpu()
        .numpy()
    )
    return image_bbox_quality(image, bbox)


def evaluate(model, dataset, class_subset, quiet=True):
    un_augment = UnAugment()
    length = len(dataset)

    n_gth = defaultdict(int)
    predictions = defaultdict(list)

    model.eval()
    model = model.cuda()

    for i, file in tqdm(enumerate(dataset.names), total=length, disable=quiet):
        try:
            image, targets = dataset[i]
        except ValueError:
            continue
        image = image.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            detections: torch.Tensor = model(image.unsqueeze(0))
        for j, name in enumerate(dataset.labels):
            if name not in class_subset:
                continue  # skip classes
            truths = targets[targets[:, 4] == j]
            n_gth[name] += len(truths)
            dets = detections[0, j + 1, ...]  # 0 is bkg_label
            mask = dets[:, 0].gt(0.0)
            if mask.any():
                dets = dets[mask, ...]
                sorted_conf, order_idx = dets[:, 0].sort(descending=True)
                tps = calculate_true_positives(dets[order_idx, 1:], truths[:, :4])
                localizations = calculate_positive_localizations(
                    dets[order_idx, 1:],
                    targets[: ,:4],
                )
                sharpness = calculate_detection_sharpness(
                    dets[order_idx, 1:], un_augment(image)
                )
                file_dict = {
                    'tps': tps,
                    'localizations': localizations,
                    'sharpness': sharpness,
                    'confs': sorted_conf,
                    'file': file,
                    'truth_idx': truths[:, 5],
                }
                predictions[name].append(file_dict)

    metrics = {}
    for name in class_subset:
        matched_gt = torch.cat([p['tps'] for p in predictions[name]], dim=0)
        true_pos = matched_gt.ge(0)
        confs = torch.cat([p['confs'] for p in predictions[name]], dim=0)
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
    gt_table = []
    fn_table = []
    detection_table = {'sharpness': [], 'confidence': [], 'tp': [], 'gt_match': []}
    for cls in predictions.values():
        for example in cls:
            tp = example['tps'].ge(0)

            detection_table['sharpness'].extend(example['sharpness'])
            detection_table['confidence'].extend(example['confs'].tolist())
            detection_table['tp'].extend(tp.tolist())
            detection_table['gt_match'].extend(example['localizations'].tolist())

            gt_id = example['tps'][tp]
            file = example['file']
            if not example['truth_idx'].nelement():
                continue
            tps = example['truth_idx'][gt_id]
            for conf, gid in zip(example['confs'][tp], tps):
                gt_table.append([file, gid.int().item(), conf.item()])

            fn = set(example['truth_idx'].tolist()) - set(tps.tolist())
            fn_table.extend([[file, idx] for idx in fn])

    metrics['gt_table'] = gt_table
    metrics['fn_table'] = fn_table
    metrics['detection_table'] = detection_table
    return metrics
