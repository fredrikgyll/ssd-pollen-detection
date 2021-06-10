import pickle
from collections import namedtuple
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.functional import to_pil_image

from model.priors.priors import PriorBox
from model.ssd import make_ssd
from model.utils.augmentations import DeNormalize, SSDAugmentation
from model.utils.data import AstmaDataset 
from model.utils.geometry import jaccard, point_form

model_name = '2021-04-28T13-02-11'
model_ch = Path('/Users/fredrikg/Projects/pollendb1/saves') / f'{model_name}.pth'


def run_model():
    from torchsummary import summary

    model = make_ssd(phase='train')
    summary(model, (3, 300, 300), 32)
    x = torch.randn(2, 3, 300, 300)
    l, c, p = model(x)
    print(l.shape, c.shape, p.shape)


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


def infer():
    import matplotlib.pyplot as plt

    from model.utils.annotate import annotate_detection

    root = Path('/Users/fredrikg/Pictures/pollen_astma/')
    transform = SSDAugmentation(train=False)
    dataset = AstmaDataset(root, 'balanced', 'test', transform)

    state = torch.load(model_ch, map_location=torch.device('cpu'))

    cfg = dict(
        size=state['size'],
        num_classes=state['num_classes'],
        layer_activation=state['layer_activations'],
        default_boxes=state['default_boxes'],
        variances=[0.1, 0.2],
    )
    model = make_ssd(phase='test', backbone=state.get('backbone', 'resnet34'), cfg=cfg)

    model.load_state_dict(state['model_state_dict'], strict=True)

    model.eval()
    for i, name in tqdm(enumerate(dataset.images), total=len(dataset)):
        try:
            img, targets = dataset[i]
        except ValueError:
            continue
        with torch.no_grad():
            detections = model(img.unsqueeze(0))

        annotate_detection(
            img,
            targets[:, :5],
            detections,
            dataset.labels,
            save=f'tmp/detections/{name[:-4]}.png',
        )


def test_decoder():
    def match(defaults, truths, threshhold: float = 0.5):
        overlap = jaccard(point_form(defaults), truths)
        # best_default, best_defailt_idx = torch.max(overlap, 0, keepdim=True)
        # matches = overlap.ge(best_default) | overlap.ge(threshhold)
        # return matches
        #  element i in matched_targets is idx of matched truth box
        matched_targets = torch.full((defaults.size(0),), -1)

        # highest jaccard overlap for every default box
        best_target, best_target_idx = overlap.max(1)
        best_default, best_default_idx = overlap.max(0)

        defaults_mask = best_target > threshhold

        matched_targets[defaults_mask] = best_target_idx[defaults_mask]
        matched_targets.index_copy_(
            0, best_default_idx, torch.arange(0, truths.size(0))
        )
        return matched_targets

    import matplotlib.pyplot as plt

    root = Path('/Users/fredrikg/Pictures/pollen_astma/')
    transform = SSDAugmentation(train=False)
    dataset = AstmaDataset(root, 'balanced', 'train', transform)

    model = make_ssd(phase='train', num_classes=len(dataset.labels) + 1)
    defaults = model.priors.transpose(0, 1)

    total_matches = torch.zeros(defaults.size(0)).int()
    bins = [0, 5776, 2166, 600, 150, 36, 4]
    steps = np.cumsum(bins)
    model.eval()
    for i in tqdm(range(len(dataset))):
        try:
            _, truth = dataset[i]
        except ValueError:
            continue
        matches = match(defaults, truth[:, :4], 0.5)
        total_matches += matches.ge(0).int()

    totals = [
        total_matches[n:m].sum().item() for n, m in list(zip(steps[:-1], steps[1:]))
    ]

    print(totals)


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
    model = make_ssd()

    img, boxes, label = trans(img, boxes, torch.ones(boxes.size(0)))
    img, *_ = denorm(img, None, None)
    target_box, target_label = encode(model.priors, boxes, label)
    pos = target_label > 0
    target_box = point_form(target_box.T[pos])

    defaults = point_form(model.priors[pos])

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
    model = make_ssd()
    criterion = MultiBoxLoss(model.priors, batch_size)
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
        tb, tl = encode(model.priors, b, la)
        ims.append(i)
        boxes.append(tb)
        labels.append(tl)
    images = torch.stack(ims, dim=0)
    gloc = torch.stack(boxes, dim=0)
    print(gloc.size())
    glabel = torch.stack(labels, dim=0).long()

    ploc, pconf = model(images)

    loss = criterion(ploc, pconf, gloc, glabel)
    print(loss.item())


if __name__ == "__main__":
    # show_img('0090_108.jpg')
    # data_pipeline()
    # run_model()
    infer()  # '0034_076.jpg' '0063_131.jpg'
    # run_model()
    # test_decoder()
    # test_encoder('0056_112.jpg')
    # test_crit(['0114_171.jpg', '0403_055.jpg', '0090_108.jpg'])
    # evaluate_model()
