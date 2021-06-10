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
from model.utils.evaluation import calculate_true_positives, evaluate, interpolate


def calc_dice(metrics, class_subset):
    vals = []
    for k in class_subset:
        tp = metrics[k]['tp']
        fp = metrics[k]['fp']
        fn = metrics[k]['ground_truths'] - tp
        vals.append( tp / (tp + 0.5 * (fp + fn)))
    return np.mean(vals)


def calc_map(metrics, class_subset):
    aps = [metrics[cls]['average_precision'] for cls in class_subset]
    return np.mean(aps)


def load_run_metrics(root: Path):
    files = sorted(root.glob('*.pkl'))
    metrics = [pickle.loads(f.read_bytes()) for f in files]
    return metrics


def run_metrics(checkpoints, datasets, class_subset, quiet=True):
    points = [int(f.stem.lstrip('ssd_')) for f in checkpoints]

    init_state = torch.load(checkpoints[0], map_location=torch.device('cuda'))

    cfg = dict(
        size=init_state['size'],
        num_classes=init_state['num_classes'],
        layer_activation=init_state['layer_activations'],
        default_boxes=init_state['default_boxes'],
        variances=[0.1, 0.2],
    )
    model = make_ssd(
        phase='test', backbone=init_state.get('backbone', 'resnet34'), cfg=cfg
    )
    model = model.cuda()

    metrics = []
    for i, file in enumerate(checkpoints):
        print(f'[{i+1:2d}/{len(checkpoints)}] {file.stem}')
        state = torch.load(file, map_location=torch.device('cuda'))
        model.load_state_dict(state['model_state_dict'], strict=True)
        metrics.append(
            [evaluate(model, dst, class_subset, quiet=quiet) for dst in datasets]
        )

    metrics = list(zip(*metrics))
    return points, metrics


def evaluate_run(run_dir: Path, data_root: Path, class_subset, quiet=True):
    checkpoints = list(run_dir.glob('ssd_*.pth'))

    transforms = SSDAugmentation(train=False)
    test_dataset = HDF5Dataset(data_root, 'balanced1', 'test', transforms)
    train_dataset = HDF5Dataset(data_root, 'balanced1', 'train', transforms)

    datasets = [test_dataset, train_dataset]

    iteration, metrics = run_metrics(checkpoints, datasets, class_subset, quiet=quiet)
    points = np.array(iteration)
    maps = [[calc_map(m, class_subset) for m in run] for run in metrics]
    maps = [np.array(m) for m in maps]

    order_idx = np.argsort(points)
    return points[order_idx], maps[0][order_idx], maps[1][order_idx]


if __name__ == "__main__":
    CLASSES = ['poaceae', 'corylus', 'alnus']

    parser = argparse.ArgumentParser(description='Evaluate SSD300 model')

    parser.add_argument(
        '--checkpoint', '-c', type=Path, help='Path to model checkpoint'
    )
    parser.add_argument('--data', '-d', type=Path, help='path to data directory')
    parser.add_argument('--mode', '-m', type=str, help='dataset mode', default='test')
    parser.add_argument(
        '--output',
        '-o',
        type=Path,
        default=Path('./metrics.pkl'),
        help='Path to save evaluation',
    )

    args = parser.parse_args()

    transforms = SSDAugmentation(train=False)
    dataset = HDF5Dataset(args.data, 'balanced1', args.mode, transforms)

    state = torch.load(args.checkpoint, map_location=torch.device('cuda'))

    cfg = dict(
        size=state['size'],
        num_classes=state['num_classes'],
        layer_activation=state['layer_activations'],
        default_boxes=state['default_boxes'],
        variances=[0.1, 0.2],
    )
    model = make_ssd(phase='test', backbone=state.get('backbone', 'resnet34'), cfg=cfg)
    model = model.cuda()

    model.load_state_dict(state['model_state_dict'], strict=True)

    metrics = evaluate(model, dataset, CLASSES, quiet=False)
    args.output.write_bytes(pickle.dumps(metrics))
    print(f'mAP: {calc_map(metrics, CLASSES):.2%}')
