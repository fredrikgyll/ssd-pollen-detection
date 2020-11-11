import torch
import numpy as np


def run_model():
    from ssd import make_ssd
    from utils import decode

    m = make_ssd()
    x = torch.randn(1, 3, 300, 300)
    l, c = m(x)
    print(decode(m.priors, l.abs(), c, soft=False))


def run_subsampler(name):
    from PIL import Image
    from pathlib import Path
    import pickle
    import numpy as np
    from utils.augmentations import SubSample

    root = Path('/Users/fredrikg/Projects/pollendb1/data')
    trf = root / 'annotations/train_labels.pkl'
    training_boxes = pickle.load(trf.open('rb'))
    im = np.array(Image.open(root / 'train' / name))  # (512, 640, 3)
    im = im.transpose(2, 0, 1)  # (3, 512, 640)
    boxes = training_boxes[name]
    labels = np.ones(len(boxes))

    sub = SubSample(im.shape[1], im.shape[0])
    sub_img, new_boxes, new_labels = sub(im, boxes, labels)
    return sub_img, new_boxes, new_labels


def patch_boxes(boxes):
    return np.hstack((boxes[:, :2], boxes[:, 2:] - boxes[:, :2]))


def show_img(name):
    import pickle
    from pathlib import Path
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import utils.augmentations as aug

    p = Path('/Users/fredrikg/Projects/pollendb1/data/train')
    trf = Path('/Users/fredrikg/Projects/pollendb1/data/annotations/train_labels.pkl')
    train_labels = pickle.load(trf.open('rb'))
    sub = aug.SubSample(640, 512)
    distort = aug.TransformerSequence(
        aug.HorizontalFlip(),
        aug.ChannelSuffle(),
    )
    pre = aug.TransformerSequence(
        aug.FromIntToFloat(),
        aug.SubtractMean(),
        aug.ToStandardForm(),
    )

    boxes = train_labels[name]
    labels = np.ones(len(boxes))
    im = np.array(Image.open(p / name))

    tim, boxes, labels = pre(im, boxes, labels)
    subim, subboxes, sublabels = sub(tim / 255, boxes, labels)
    fim, fboxes, _ = distort(subim, subboxes, sublabels)
    print(subboxes, fboxes)
    subim = subim.transpose(1, 2, 0)
    fim = fim.transpose(1, 2, 0)
    subboxes = np.clip(patch_boxes(subboxes), 0, 1) * 300
    fboxes = np.clip(patch_boxes(fboxes), 0, 1) * 300
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    ax1.imshow(im, interpolation='none')
    ax1.set_title('Original')
    for *xy, w, h in patch_boxes(boxes):
        ax1.add_patch(patches.Rectangle(xy, w, h, edgecolor='green', fill=False))
    ax2.imshow(subim, interpolation='none')
    ax2.set_title('Subsample')
    for *xy, w, h in subboxes:
        ax2.add_patch(patches.Rectangle(xy, w, h, edgecolor='red', fill=False))
    ax3.imshow(fim, interpolation='none')
    ax3.set_title('Channel Shuffle')
    for *xy, w, h in fboxes:
        ax3.add_patch(patches.Rectangle(xy, w, h, edgecolor='blue', fill=False))
    plt.show()


if __name__ == "__main__":
    show_img('0090_108.jpg')
