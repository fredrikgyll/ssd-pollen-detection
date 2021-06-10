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


def calc_map(metrics, class_subset):
    aps = [metrics[cls]['average_precision'] for cls in class_subset]
    return np.mean(aps)


def load_run_metrics(root: Path):
    files = sorted(root.glob('*.pkl'))
    metrics = [pickle.loads(f.read_bytes()) for f in files]
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
        state = torch.load(file, map_location=torch.device('cuda'))
        model.load_state_dict(state['model_state_dict'], strict=True)
        metrics = [evaluate(model, dst, class_subset, quiet=quiet) for dst in datasets]
        maps.append([calc_map(m, class_subset) for m in metrics])
    test_map, train_map = list(zip(*maps))
    test_map = np.array(test_map)
    train_map = np.array(train_map)

    order_idx = np.argsort(points)
    return points[order_idx], test_map[order_idx], train_map[order_idx]


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
    model = make_ssd(phase='test', num_classes=len(dataset.labels) + 1)
    model = model.cuda()

    state = torch.load(args.checkpoint, map_location=torch.device('cuda'))
    model.load_state_dict(state['model_state_dict'], strict=True)

    metrics = evaluate(model, dataset, CLASSES, quiet=False)
    args.output.write_bytes(pickle.dumps(metrics))
