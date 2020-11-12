from random import randint
import time
import argparse
from pathlib import Path

import torch
import torch.utils.data as data
from torch.optim import SGD

from ssd import make_ssd
from loss import MultiBoxLoss
from utils.augmentations import get_transform
from utils.geometry import encode
from utils.data import Pollene1Dataset, collate

parser = argparse.ArgumentParser(description='Train SSD300 model')
parser.add_argument(
    '--cuda',
    action='store_true',
    help='Train model on cuda enabled GPU',
)
parser.add_argument(
    '--data',
    type=Path,
    help='path to data directory',
)
parser.add_argument(
    '--weights',
    type=Path,
    help='Path th VGG16 weights',
)
parser.add_argument(
    '--epochs',
    type=int,
    help='# of epochs to run',
)
parser.add_argument(
    '--workers',
    type=int,
    help='# of workers in dataloader',
)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


def train(args):
    batch_size = 8

    # Data
    root = args.data
    transforms = get_transform()
    dataset = Pollene1Dataset(root, transforms)
    data_loader = data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
    )

    # Model init
    ssd_net = make_ssd(300, 2)
    optimizer = SGD(ssd_net.parameters(), lr=1e-3, momentum=1e-1)
    criterion = MultiBoxLoss(ssd_net.priors, batch_size)
    if args.cuda:
        ssd_net = ssd_net.cuda()
        criterion = criterion.cuda()

    # Weights
    vgg_weights = torch.load(args.weights)
    ssd_net.base.load_state_dict(vgg_weights)
    ssd_net.extra.apply(weights_init)
    ssd_net.loc_head.apply(weights_init)
    ssd_net.conf_head.apply(weights_init)

    ssd_net.train()

    batch_iterator = iter(data_loader)

    for i in range(args.epochs):
        print(f'===== Iteration {i:2d} =====')
        dt0 = time.time()
        images, targets, labels = next(batch_iterator)
        dt1 = time.time()
        target_boxes = []
        target_labels = []
        for truth, label in zip(targets, labels):
            target_box, target_label = encode(ssd_net.priors, truth, label)
            target_boxes.append(target_box)
            target_labels.append(target_label)
        gloc = torch.stack(target_boxes, dim=0)
        glabel = torch.stack(target_labels, dim=0).long()
        dt2 = time.time()
        print(f"Batch load:\t{dt1-dt0:3.1f}")
        print(f"Batch encode:\t{dt2-dt1:3.1f}")
        if args.cuda:
            gloc, glabel, images = gloc.cuda(), glabel.cuda(), images.cuda()

        t0 = time.time()
        ploc, pconf = ssd_net(images)
        t1 = time.time()
        optimizer.zero_grad()

        loss = criterion(ploc, pconf, gloc, glabel)
        loss.backward()
        optimizer.step()
        t2 = time.time()
        print(f"Loss:\t{loss.item():3.1f}")
        print(f"Forward:\t{t1-t0:3.1f} sec")
        print(f"Backward:\t{t2-t1:3.1f} sec")
        print(f"Total:\t\t{t2-t0:3.1f} sec")


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
