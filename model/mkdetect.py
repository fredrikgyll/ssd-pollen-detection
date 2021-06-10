import argparse
from os import sep
from pathlib import Path

from cv2 import data
from ssd import make_ssd
import torch
from utils.augmentations import SSDAugmentation

from utils.data import Pollene1Dataset


parser = argparse.ArgumentParser(description='Train SSD300 model')

parser.add_argument('--checkpoint', '-c', type=Path, help='Path to model checkpoint')
parser.add_argument('--data', '-d', type=Path, help='path to data directory')
parser.add_argument('--output', '-o', type=Path, help='Path to save folder')
parser.add_argument(
    '--cuda', action='store_true', help='Train model on cuda enabled GPU'
)


def clean_pred(confs, predictions, dim):
    predictions *= dim
    return (
        confs.squeeze(-1).cpu().numpy(),
        predictions.int().cpu().numpy(),
    )


def clean_gt(targets, dim):
    targets[:, :4] *= dim
    return targets.int().cpu().numpy()


CLASSES = ['pollen']

if __name__ == "__main__":
    args = parser.parse_args()

    det_dir = args.output / 'detection-results'
    gt_dir = args.output / 'ground-truth'
    det_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)

    dim = 300

    model = make_ssd(phase='test')
    model_state = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(model_state, strict=True)

    transform = SSDAugmentation(train=False)
    dataset = Pollene1Dataset(args.data, 'test', transform)
    length = len(dataset)

    model.eval()

    for i in range(3):
        file = Path(dataset.images[i])
        image, targets = dataset[i]
        gt_lines = [
            f'{CLASSES[gt[4]]} {" ".join(str(x) for x in gt[:4])}'
            for gt in clean_gt(targets, dim)
        ]
        det_lines = []
        with torch.no_grad():
            detections = model(image.unsqueeze(0))
            for j, name in enumerate(CLASSES, start=1):
                dets = detections[0, j, ...]  # only one class which is nr. 1
                mask = dets[:, 0].gt(0.0)
                if mask.any():
                    dets = dets[mask, ...]
                    confs, boxes = torch.split(dets, [1, 4], dim=1)
                    confs, boxes = clean_pred(confs, boxes, dim)
                    det_lines.extend(
                        [
                            f'{name} {conf} {" ".join(str(b) for b in bounds)}'
                            for conf, bounds in zip(confs, boxes)
                        ]
                    )
        out_name = file.stem + '.txt'
        gt_file = gt_dir / out_name
        det_file = det_dir / out_name
        gt_file.write_text('\n'.join(gt_lines))
        det_file.write_text('\n'.join(det_lines))
