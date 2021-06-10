import pickle
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import to_pil_image

from model.priors.priors import PriorBox
from model.ssd import make_ssd
from model.utils.augmentations import DeNormalize, SSDAugmentation
from model.utils.data import AstmaDataset, Pollene1Dataset, collate

model_name = '2021-04-06T09-11-21'
model_ch = Path('/Users/fredrikg/Projects/pollendb1/saves') / f'{model_name}.pth'


def run_model():
    from model.utils import decode

    m = make_ssd()
    x = torch.randn(2, 3, 300, 300)
    l, c = m(x)
    print(decode(m.priors, l.abs(), c, soft=False))


def run_subsampler(name):
    from model.utils.augmentations import SubSample

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


def to_plt_img(image):
    return np.array(to_pil_image(image, mode='RGB'))


def show_img(name):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    import model.utils.augmentations as aug

    p = Path('/Users/fredrikg/Projects/pollendb1/data/train')
    trf = Path('/Users/fredrikg/Projects/pollendb1/data/annotations/train_labels.pkl')
    train_labels = pickle.load(trf.open('rb'))
    flip = aug.VerticalFlip(1.1)
    shuffle = aug.ColorSift(1.1)
    std_transform = aug.TransformerSequence(
        aug.ToStandardForm(),
        aug.SubSample(640, 512),
        aug.SubtractMean(),
    )
    boxes = train_labels[name]
    labels = torch.ones(len(boxes))
    im = Image.open(p / name)

    im, boxes, labels = std_transform(im, boxes, labels)
    fim, fboxes, flabels = flip(im, boxes, labels)
    shuffim, shuffboxes, _ = shuffle(fim, fboxes, flabels)
    print(im[0, :10, :10])
    dim = 300
    boxes = np.clip(patch_boxes(boxes) * dim, 0, dim)
    fboxes = np.clip(patch_boxes(fboxes) * dim, 0, dim)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

    ax1.imshow(to_plt_img(im), interpolation='none')
    ax1.set_title('Subsample')
    for *xy, w, h in boxes:
        ax1.add_patch(patches.Rectangle(xy, w, h, edgecolor='green', fill=False))

    ax2.imshow(to_plt_img(fim), interpolation='none')
    ax2.set_title('Flip')
    for *xy, w, h in fboxes:
        ax2.add_patch(patches.Rectangle(xy, w, h, edgecolor='red', fill=False))

    ax3.imshow(to_plt_img(shuffim), interpolation='none')
    ax3.set_title('Channel Shuffle')
    for *xy, w, h in fboxes:
        ax3.add_patch(patches.Rectangle(xy, w, h, edgecolor='blue', fill=False))
    plt.show()


def data_pipeline():

    root = Path('/Users/fredrikg/Projects/pollendb1/data')
    transform = SSDAugmentation()
    dataset = Pollene1Dataset(root, 'train', transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=1, collate_fn=collate
    )
    dataiter = iter(dataloader)
    image, bboxes, labels = next(dataiter)
    print(image.shape)
    print(bboxes.shape)
    print(labels.shape)


def infer(name):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    from model.utils.augmentations import DeNormalize, get_transform
    from model.utils.geometry import decode

    dim = 300
    model = make_ssd()
    model_state = torch.load(model_ch, map_location=torch.device('cpu'))
    model.load_state_dict(model_state, strict=False)
    p = Path('/Users/fredrikg/Projects/pollendb1/data/test')
    trf = Path('/Users/fredrikg/Projects/pollendb1/data/annotations/test.pkl')
    train_labels = pickle.load(trf.open('rb'))
    boxes = train_labels[name]
    labels = torch.ones(len(boxes))
    img = Image.open(p / name)

    denorm = DeNormalize()
    transform = get_transform(train=False)

    img, boxes, labels = transform(img, boxes, labels)
    model.eval()
    with torch.no_grad():
        ploc, pconf, _ = model(img.unsqueeze(0))
        print('done out')
        out_boxes = decode(model.priors, ploc, pconf, [0.1, 0.2])[0]
        print(out_boxes)
    img, *_ = denorm(img, boxes, labels)
    boxes = np.clip(patch_boxes(boxes.numpy()) * dim, 0, dim)
    out_boxes = np.clip(patch_boxes(out_boxes[0].numpy()) * dim, 0, dim)

    _, ax = plt.subplots()
    ax.imshow(to_plt_img(img), interpolation='none')
    # ax.set_title('Subsample')
    for *xy, w, h in boxes:
        ax.add_patch(patches.Rectangle(xy, w, h, edgecolor='green', fill=False))
    for *xy, w, h in out_boxes:
        ax.add_patch(patches.Rectangle(xy, w, h, edgecolor='red', fill=False))
    ax.set_axis_off()
    plt.show()


def infer2():
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    from model.utils.augmentations import DeNormalize

    root = Path('/Users/fredrikg/Pictures/pollen_astma/')
    transform = SSDAugmentation(train=False)
    dataset = AstmaDataset(root, 'test', transform)
    colors = {
        'poaceae': (0.890625, 0.101562, 0.101562),
        'corylus': (0.215686, 0.494118, 0.721569),
        'alnus': (0.301961, 0.686275, 0.290196),
        'misc': (0.596078, 0.305882, 0.639216),
        'unknown': (0.54902, 0.337255, 0.294118),
    }

    dim = 300
    model = make_ssd(phase='test', num_classes=len(dataset.labels) + 1)
    model_state = torch.load(model_ch, map_location=torch.device('cpu'))
    model.load_state_dict(model_state, strict=False)

    lens = [len(dataset.bboxes[n]) for n in dataset.images]
    sorted = np.argsort(lens)[::-1]

    img, targets = dataset[sorted[0]]
    boxes, labels = targets[:, :4], targets[:, 4]
    denorm = DeNormalize()
    print(f'targets>\n{boxes}')
    model.eval()
    with torch.no_grad():
        detections = model(img.unsqueeze(0))
    out = []
    for i in [1, 2, 3]:
        dets = detections[0, i, ...]  # only one class which is nr. 1
        mask = dets[:, 0].gt(0.0)
        dets = dets[mask, ...]
        out.append(dets.data)

    img, _, labels = denorm(img, boxes, labels)
    print(f'predictions>\n{out}')
    boxes = np.clip(patch_boxes(boxes.numpy()) * dim, 0, dim)
    out_bbox = [np.clip(patch_boxes(b[:, 1:].numpy()) * dim, 0, dim) for b in out]
    out_conf = [b[:, 0].numpy() for b in out]

    _, ax = plt.subplots()
    ax.imshow(to_plt_img(img), interpolation='none')
    # ax.set_title('Subsample')

    offsets = {
        'gt': lambda xy, w, h: (xy[0] + (w // 2), xy[1] + (h // 2)),
        'poaceae': lambda xy, w, h: (xy[0], xy[1] + h + 10),
        'corylus': lambda xy, w, h: (xy[0] + w, xy[1]),
        'alnus': lambda xy, w, h: (xy[0] + w, xy[1] + h + 10),
    }

    for (*xy, w, h), l in zip(boxes, labels.int().tolist()):
        ax.add_patch(patches.Rectangle(xy, w, h, edgecolor='green', fill=False))
        ax.text(*offsets['gt'](xy, w, h), dataset.labels[l], color='green', fontsize=10)
    for lab, boxes, confs in zip(dataset.labels, out_bbox, out_conf):
        for (*xy, w, h), c in zip(boxes, confs):
            if not confs.size:
                continue
            ax.add_patch(patches.Rectangle(xy, w, h, edgecolor=colors[lab], fill=False))
            ax.text(
                *offsets[lab](xy, w, h),
                f'{lab} {c:.2f}',
                color=colors[lab],
                fontsize=10,
            )
    ax.set_axis_off()
    plt.show()


def test_encoder(name):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    from model.utils import encode
    from model.utils.geometry import point_form

    p = Path('/Users/fredrikg/Projects/pollendb1/data/train')
    trf = Path('/Users/fredrikg/Projects/pollendb1/data/annotations/train.pkl')
    train_labels = pickle.load(trf.open('rb'))
    boxes = train_labels[name]
    img = Image.open(p / name)
    trans = SSDAugmentation(train=False)
    denorm = DeNormalize()
    m = make_ssd()

    img, boxes, label = trans(img, boxes, torch.ones(boxes.size(0)))
    img, *_ = denorm(img, None, None)
    target_box, target_label = encode(m.priors, boxes, label)
    pos = target_label > 0
    target_box = point_form(target_box.T[pos])

    defaults = point_form(m.priors[pos])

    chosen, chosen_idx = target_label.sort(descending=True)
    cpos = chosen > 0
    print(chosen_idx[cpos])
    dim = 300
    out_boxes = np.clip(patch_boxes(defaults) * dim, 0, dim)
    boxes = np.clip(patch_boxes(boxes) * dim, 0, dim)

    _, ax = plt.subplots()
    ax.imshow(to_plt_img(img), interpolation='none')
    for *xy, w, h in out_boxes:
        ax.add_patch(
            patches.Rectangle(xy, w, h, edgecolor='red', fill=False, alpha=0.6)
        )
    for *xy, w, h in boxes:
        ax.add_patch(
            patches.Rectangle(xy, w, h, edgecolor='green', linewidth=1, fill=False)
        )
    ax.set_axis_off()
    plt.show()


def test_crit(names):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    from model.loss import MultiBoxLoss
    from model.utils import encode
    from model.utils.augmentations import get_transform
    from model.utils.geometry import point_form

    batch_size = len(names)
    ssd_net = make_ssd()
    criterion = MultiBoxLoss(ssd_net.priors, batch_size)
    p = Path('/Users/fredrikg/Projects/pollendb1/data/train')
    trf = Path('/Users/fredrikg/Projects/pollendb1/data/annotations/train_bboxes.pkl')
    train_labels = pickle.load(trf.open('rb'))
    trans = get_transform(train=False)
    ims = []
    boxes = []
    labels = []
    for n in names:
        i, b, la = trans(
            Image.open(p / n),
            train_labels[n],
            torch.ones(train_labels[n].size(0)),
        )
        tb, tl = encode(ssd_net.priors, b, la)
        ims.append(i)
        boxes.append(tb)
        labels.append(tl)
    images = torch.stack(ims, dim=0)
    gloc = torch.stack(boxes, dim=0)
    print(gloc.size())
    glabel = torch.stack(labels, dim=0).long()

    ploc, pconf = ssd_net(images)

    loss = criterion(ploc, pconf, gloc, glabel)
    print(loss.item())


if __name__ == "__main__":
    # show_img('0090_108.jpg')
    # data_pipeline()
    # run_model()
    infer2()  # '0034_076.jpg' '0063_131.jpg'
    # test_encoder('0056_112.jpg')
    # test_crit(['0114_171.jpg', '0403_055.jpg', '0090_108.jpg'])
    # evaluate_model()
