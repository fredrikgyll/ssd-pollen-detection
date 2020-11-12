import io
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import numpy as np
from pathlib import Path
from PIL import Image
import pickle


def run_model():
    from ssd import make_ssd
    from utils import decode

    m = make_ssd()
    x = torch.randn(1, 3, 300, 300)
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
    dataset = Pollene1Dataset(root, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=1, collate_fn=collate
    )
    dataiter = iter(dataloader)
    image, bboxes, labels = next(dataiter)
    print(image.shape)
    print(bboxes.shape)
    print(labels.shape)


def infer(name):
    from ssd import make_ssd
    from utils.geometry import decode
    from utils.augmentations import get_transform, DeNormalize
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    dim = 300
    model_ch = Path(
        '/Users/fredrikg/Projects/pollendb1/saves/2020-11-12T17-43-58/ssd_epoch_19.pth'
    )
    model = make_ssd()
    model_state = torch.load(model_ch, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)

    p = Path('/Users/fredrikg/Projects/pollendb1/data/train')
    trf = Path('/Users/fredrikg/Projects/pollendb1/data/annotations/train_bboxes.pkl')
    train_labels = pickle.load(trf.open('rb'))
    boxes = train_labels[name]
    labels = torch.ones(len(boxes))
    img = Image.open(p / name)

    denorm = DeNormalize()
    transform = get_transform(train=False)

    img, boxes, labels = transform(img, boxes, labels)
    img, _, _ = denorm(img, boxes, labels)
    model.eval()
    with torch.no_grad():
        ploc, pconf = model(img.unsqueeze(0))
        print('done out')
        out_boxes = decode(model.priors, ploc, pconf, soft=False, iou_thr=0.3)

    boxes = np.clip(patch_boxes(boxes.numpy()) * dim, 0, dim)
    out_boxes = np.clip(patch_boxes(out_boxes.numpy()) * dim, 0, dim)

    _, ax = plt.subplots()
    ax.imshow(to_plt_img(img), interpolation='none')
    ax.set_title('Subsample')
    for *xy, w, h in boxes:
        ax.add_patch(patches.Rectangle(xy, w, h, edgecolor='green', fill=False))
    for *xy, w, h in out_boxes:
        ax.add_patch(patches.Rectangle(xy, w, h, edgecolor='red', fill=False))
    plt.show()


if __name__ == "__main__":
    # show_img('0090_108.jpg')
    # data_pipeline()
    # run_model()
    infer('0090_108.jpg')
