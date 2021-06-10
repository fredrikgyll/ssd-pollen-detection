from collections import namedtuple
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import to_pil_image
import numpy as np
from pathlib import Path
from PIL import Image
import pickle
from evaluate import evaluate
from ssd import make_ssd

from utils.augmentations import DeNormalize, get_transform
from utils.data import Pollene1Dataset, collate


def run_model():
    from ssd import make_ssd
    from utils import decode

    m = make_ssd()
    x = torch.randn(2, 3, 300, 300)
    l, c = m(x)
    print(decode(m.priors, l.abs(), c, soft=False))


def run_subsampler(name):
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


def to_plt_img(image):
    return np.array(to_pil_image(image, mode='RGB'))


def show_img(name):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import utils.augmentations as aug

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
    from utils.data import Pollene1Dataset, collate
    from utils.augmentations import get_transform

    root = Path('/Users/fredrikg/Projects/pollendb1/data')
    transform = get_transform()
    dataset = Pollene1Dataset(root, 'train', transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=1, collate_fn=collate
    )
    dataiter = iter(dataloader)
    image, bboxes, labels = next(dataiter)
    print(image.shape)
    print(bboxes.shape)
    print(labels.shape)


def evaluate_model():
    import matplotlib.pyplot as plt

    model_ch = Path('/Users/fredrikg/Projects/pollendb1/saves/2020-11-13T21-10-01.pth')
    model = make_ssd()
    model_state = torch.load(model_ch, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)

    root = Path('/Users/fredrikg/Projects/pollendb1/data')

    transforms = get_transform(train=False)
    dataset = Pollene1Dataset(root, 'test', transforms)
    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
        pin_memory=True,
    )
    args = namedtuple('args', ['cuda'])
    args.cuda = False
    p, r = evaluate(model, data_loader, args)
    _, ax = plt.subplots()
    ax.plot(r, p)
    plt.show()


def infer(name):
    from ssd import make_ssd
    from utils.geometry import decode
    from utils.augmentations import get_transform, DeNormalize
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    dim = 300
    model_ch = Path('/Users/fredrikg/Projects/pollendb1/saves/2020-11-16T18-43-31.pth')
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
        ploc, pconf = model(img.unsqueeze(0))
        print('done out')
        out_boxes = decode(model.priors, ploc, pconf)[0]
    img, *_ = denorm(img, boxes, labels)
    boxes = np.clip(patch_boxes(boxes.numpy()) * dim, 0, dim)
    out_boxes = np.clip(patch_boxes(out_boxes[0].numpy()) * dim, 0, dim)

    _, ax = plt.subplots()
    ax.imshow(to_plt_img(img), interpolation='none')
    ax.set_title('Subsample')
    for *xy, w, h in boxes:
        ax.add_patch(patches.Rectangle(xy, w, h, edgecolor='green', fill=False))
    for *xy, w, h in out_boxes:
        ax.add_patch(patches.Rectangle(xy, w, h, edgecolor='red', fill=False))
    plt.show()


def test_encoder(name):
    from ssd import make_ssd
    from utils import encode
    from utils.augmentations import get_transform
    from utils.geometry import point_form
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    p = Path('/Users/fredrikg/Projects/pollendb1/data/train')
    trf = Path('/Users/fredrikg/Projects/pollendb1/data/annotations/train_bboxes.pkl')
    train_labels = pickle.load(trf.open('rb'))
    boxes = train_labels[name]
    img = Image.open(p / name)
    trans = get_transform(train=False)
    denorm = DeNormalize()
    m = make_ssd()

    img, boxes, label = trans(img, boxes, torch.ones(boxes.size(0)))
    img, *_ = denorm(img, None, None)
    target_box, target_label = encode(m.priors, boxes, label)
    pos = target_label > 0
    target_box = point_form(target_box.T[pos])

    defaults = point_form(m.priors)[: 773 : 19 * 19, :]

    chosen, chosen_idx = target_label.sort(descending=True)
    cpos = chosen > 0
    print(chosen_idx[cpos])
    dim = 300
    out_boxes = np.clip(patch_boxes(defaults.numpy()) * dim, 0, dim)
    boxes = np.clip(patch_boxes(boxes) * dim, 0, dim)

    _, ax = plt.subplots()
    ax.imshow(to_plt_img(img), interpolation='none')
    ax.set_title('Subsample')
    for *xy, w, h in out_boxes:
        ax.add_patch(
            patches.Rectangle(xy, w, h, edgecolor='red', fill=False, alpha=0.9)
        )
    for *xy, w, h in boxes:
        ax.add_patch(
            patches.Rectangle(xy, w, h, edgecolor='green', linewidth=3, fill=False)
        )
    plt.show()


def test_crit(names):
    from ssd import make_ssd
    from loss import MultiBoxLoss
    from utils import encode
    from utils.augmentations import get_transform
    from utils.geometry import point_form
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

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
    infer('0038_069.jpg')  # '0114_171.jpg')
    # test_crit(['0114_171.jpg', '0403_055.jpg', '0090_108.jpg'])
    # evaluate_model()
