import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from cv2 import data
from icecream import ic
from tqdm.auto import tqdm

from model.ssd import make_ssd
from model.utils.augmentations import SSDAugmentation
from model.utils.data import HDF5Dataset
from model.utils.geometry import jaccard


def interpolate(precision, recall):
    """Return list of 11-point interpolation points
    :param precision: precision by rising recall
    :param recall: monotonically increasing recall levels
    :returns: list of 11-point interpolation
    """
    pre = np.append(precision, 0.0)
    rec = np.append(recall, 1.0)
    interpolation = []
    for p in np.linspace(0, 1, 11):
        interpolation.append(np.max(pre[rec >= p]))
    return np.array(interpolation)


def evaluate(model, dataset, class_subset, cuda=False, quiet=True):
    length = len(dataset)

    n_gth = defaultdict(int)
    predictions = defaultdict(list)
    confidences = defaultdict(list)

    model.eval()
    if cuda:
        model = model.cuda()

    for i in tqdm(range(length), disable=quiet):
        file = dataset.names[i]
        try:
            image, targets = dataset[i]
        except ValueError:
            print('error:', i, file)
            continue
        with torch.no_grad():
            if cuda:
                image = image.cuda()
                targets = targets.cuda()
            detections = model(image.unsqueeze(0))
            for j, name in enumerate(dataset.labels):
                if name not in class_subset:
                    continue  # skip classes
                truth = targets[targets[:, 4] == j, :4]
                n_gth[name] += len(truth)
                dets = detections[0, j + 1, ...]  # 0 is bkg_label
                mask = dets[:, 0].gt(0.0)
                if mask.any():
                    dets = dets[mask, ...]
                    sorted_conf, order_idx = dets[:, 0].sort(descending=True)
                    iou = jaccard(dets[order_idx, 1:], truth)
                    preds = torch.zeros(dets.size(0), dtype=int)

                    for ground_column in iou.split(1, 1):
                        tp = torch.nonzero(ground_column.squeeze(-1) > 0.5).squeeze(-1)
                        if tp.nelement() > 0:
                            preds[tp[0]] = 1
                    predictions[name].append(preds)
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
            'fp': (preds == 0).sum().item(),
        }

    return metrics


if __name__ == "__main__":
    CLASSES = ['poaceae', 'corylus', 'alnus', 'unknown']

    parser = argparse.ArgumentParser(description='Evaluate SSD300 model')

    parser.add_argument(
        '--checkpoint', '-c', type=Path, help='Path to model checkpoint'
    )
    parser.add_argument('--data', '-d', type=Path, help='path to data directory')
    parser.add_argument(
        '--output',
        '-o',
        type=Path,
        detault=Path('./metrics.plk'),
        help='Path to save evaluation',
    )
    parser.add_argument(
        '--cuda', action='store_true', help='Train model on cuda enabled GPU'
    )

    args = parser.parse_args()

    transforms = SSDAugmentation(train=False)
    dataset = HDF5Dataset(args.data, 'balanced1', 'test', transforms)

    model = make_ssd(phase='test', num_classes=len(dataset.labels) + 1)
    model_state = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(model_state, strict=True)

    metrics = evaluate(model, dataset, CLASSES, cuda=args.cuda, quiet=False)

    args.output.write_bytes(pickle.dumps(metrics))
