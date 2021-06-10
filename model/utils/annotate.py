from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as f
from icecream import ic

from model.utils.evaluation import calculate_true_positives


def _points(p_min, p_max):
    return tuple(p_min), p_max[0], p_max[1]


def _label_offset(box, p, w, h, dx=0, dy=0):
    x, y = p
    if box['gt']:
        y += h // 2
    return (x + dx, y + dy)


def _mk_box(*, target=None, detection=None, label=None, cls=None, dim=300, tp=False):
    if target is not None:
        t = target.mul(dim).int()
        box = t[:2].tolist(), (t[2:4] - t[:2]).tolist()
        lab_idx = target[4].int().item()
        text = cls[lab_idx][:3]
        conf = None
        key = 'gt'
    else:
        conf = detection[0].item()
        t = detection.mul(dim).int()
        box = t[1:3].tolist(), (t[3:] - t[1:3]).tolist()
        lab_idx = label
        text = f'{cls[lab_idx][:3]} {conf:.0%}'
        key = 'tp' if tp else cls[lab_idx]
    return {'box': box, 'label': text, 'conf': conf, 'key': key}


def annotate_detection(image, targets, detections, class_list, name='', save=None):
    dim = 300
    norm_alpha = [-2.11790393, -2.03571429, -1.80444444]  # -mean/std
    norm_beta = [4.36681223, 4.46428571, 4.44444444]  # 1/std
    norm_img = f.normalize(image, norm_alpha, norm_beta, inplace=True)
    img = norm_img.permute(1, 2, 0).clamp(0, 1).numpy()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (dim,dim))
    boxes = [_mk_box(target=b, cls=class_list, dim=dim) for b in targets]
    for i in range(len(class_list)):
        dets = detections[0, i + 1, ...]  # only one class which is nr. 1
        mask = dets[:, 0].gt(0.0)
        dets = dets[mask]
        order_idx = torch.argsort(dets[:, 0], descending=True)
        tps = calculate_true_positives(
            dets[order_idx, 1:], targets[targets[:, 4] == i, :4]
        ).ge(0)
        boxes.extend(
            [
                _mk_box(detection=d, label=i, cls=class_list, dim=dim, tp=tp)
                for d, tp in zip(dets[order_idx], tps)
            ]
        )

    offsets = {
        'gt': lambda xy, w, h: (xy[0] + (w // 2), xy[1] + (h // 2)),
        'poaceae': lambda xy, w, h: (xy[0], xy[1] + h + 10),
        'corylus': lambda xy, w, h: (xy[0] + w, xy[1] - 10),
        'alnus': lambda xy, w, h: (xy[0] + w, xy[1] + h + 10),
    }
    colors = {
        'poaceae': '#ff6eb4',
        'corylus': '#ff6eb4',
        'alnus': '#ff6eb4',
        'gt': '#228b22',
        'tp': '#cd2626',
    }
    text_conf = {
        'horizontalalignment': 'center',
        'verticalalignment': 'center',
        'size': 10,
        # 'bbox': text_box
    }
    rect_conf = {'fill': False, 'lw': 1.0, 'ls': '-'}

    fig = plt.figure(figsize=(4, 4))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)

    for box in boxes:
        color = colors[box['key']]
        plt_point = _points(*box['box'])
        ax.add_patch(patches.Rectangle(*plt_point, color=color, **rect_conf))
        if box['key'] != 'tp':
            org = offsets[box['key']](*plt_point)
            ax.text(*org, box['label'], color=color, **text_conf)
    if save:
        plt.savefig(save, dpi=100)
        plt.close()
    else:
        plt.show()


def _ann_box(annotation, class_list):
    bbox = annotation[:4]
    box = bbox[:2], bbox[2:] - bbox[:2]
    return {'box': box, 'cls': class_list[annotation[4]]}


def annotate_gt(ax, img, annotations, class_list, with_labels=False):
    boxes = [_ann_box(b, class_list) for b in annotations]

    colors = {
        'poaceae': '#1f77b4',
        'corylus': '#ff7f0e',
        'alnus': '#2ca02c',
    }
    text_conf = {
        'horizontalalignment': 'center',
        'verticalalignment': 'center',
        'size': 10,
        # 'bbox': text_box
    }
    rect_conf = {'fill': False, 'lw': 0.2, 'ls': '-'}
    ax.imshow(img)
    for box in boxes:
        c = colors[box['cls']]
        plt_point = _points(*box['box'])
        ax.add_patch(patches.Rectangle(*plt_point, color=c, **rect_conf))
        if with_labels:
            ax.text(plt_point[0], box['cls'], color=c, **text_conf)
