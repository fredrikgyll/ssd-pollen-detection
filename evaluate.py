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


def calc_map(metrics, class_subset):
    aps = ap = [metrics[cls]['average_precision'] for cls in class_subset]
    return np.mean(aps)


def load_run_metrics(root: Path):
    files = sorted(root.glob('*.pkl'))
    metrics = [pickle.loads(f.read_bytes()) for f in files]
    return metrics


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
            'fp': (true_pos == 0).sum().item(),
        }

    return metrics


def evaluate_run(run_dir: Path, data_root: Path, class_subset, quiet=True):
    checkpoints = list(run_dir.glob('ssd_*.pth'))
    points = np.array([int(f.stem.lstrip('ssd_')) for f in checkpoints])

    maps = []

    transforms = SSDAugmentation(train=False)
    test_dataset = HDF5Dataset(data_root, 'balanced1', 'test', transforms)
    train_dataset = HDF5Dataset(data_root, 'balanced1', 'train', transforms)

    datasets = [test_dataset, train_dataset]

    model = make_ssd(phase='test', num_classes=len(train_dataset.labels) + 1)
    model = model.cuda()

    for i, file in enumerate(checkpoints):
        not quiet and print(f'[{i+1:2d}/{len(checkpoints)}] {file.stem}')
        model_state = torch.load(file, map_location=torch.device('cuda'))
        model.load_state_dict(model_state, strict=True)
        metrics = [evaluate(model, dst, class_subset, quiet=quiet) for dst in datasets]
        maps.append([calc_map(m, class_subset) for m in metrics])
    test_map, train_map = list(zip(*[np.array(m) for m in maps]))

    order_idx = np.argsort(points)
    return points[order_idx], test_map[order_idx], train_map[order_idx]


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
        default=Path('./metrics.pkl'),
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
