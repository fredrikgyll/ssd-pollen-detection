import argparse
import pickle
from collections import defaultdict
from os import sep
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

parser = argparse.ArgumentParser(description='Train SSD300 model')

parser.add_argument('--checkpoint', '-c', type=Path, help='Path to model checkpoint')
parser.add_argument('--data', '-d', type=Path, help='path to data directory')
parser.add_argument('--output', '-o', type=Path, help='Path to save folder')
parser.add_argument(
    '--cuda', action='store_true', help='Train model on cuda enabled GPU'
)

CLASSES = ['poaceae', 'corylus', 'alnus', 'unknown']

if __name__ == "__main__":
    args = parser.parse_args()

    n_gth = defaultdict(int)
    predictions = defaultdict(list)
    confidences = defaultdict(list)

    transforms = SSDAugmentation(train=False)
    dataset = HDF5Dataset(args.data, 'balanced1', 'test', transforms)

    model = make_ssd(phase='test', num_classes=len(dataset.labels) + 1)
    model_state = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(model_state, strict=True)

    length = len(dataset)

    model.eval()
    if args.cuda:
        model = model.cuda()

    for i in tqdm(range(length)):
        file = dataset.names[i]
        try:
            image, targets = dataset[i]
        except ValueError:
            print('error:', i, file)
            continue
        with torch.no_grad():
            if args.cuda:
                image = image.cuda()
                targets = targets.cuda()
            detections = model(image.unsqueeze(0))
            for j, name in enumerate(CLASSES):
                truth = targets[targets[:, 4] == j, :4]
                n_gth[name] += len(truth)
                dets = detections[0, j + 1, ...]  # 0 is bkg_label
                mask = dets[:, 0].gt(0.0)
                if mask.any():
                    dets = dets[mask, ...]
                    sorted_conf, order_idx = dets[:, 0].sort(descending=True)
                    iou = jaccard(dets[order_idx, 1:], truth)
                    preds = torch.zeros(dets.size(0))

                    for ground_column in iou.split(1, 1):
                        tp = torch.nonzero(ground_column.squeeze() > 0.5).squeeze(-1)
                        if tp.nelement() > 0:
                            preds[tp[0]] = 1
                    predictions[name].append(preds)
                    confidences[name].append(sorted_conf)
    out = {}
    for name in CLASSES:
        preds = torch.cat(predictions[name], dim=0)
        confs = torch.cat(confidences[name], dim=0)
        _, order = confs.sort(descending=True)
        preds = torch.cumsum(preds[order], dim=0).cpu().numpy()

        out[name] = {
            'precision': preds / np.arange(1, preds.size + 1),
            'recall': preds / n_gth[name],
        }

    (args.output / 'map.pkl').write_bytes(pickle.dumps(out))
