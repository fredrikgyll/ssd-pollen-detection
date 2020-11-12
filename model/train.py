import pickle
import time
import argparse
from pathlib import Path
import datetime
import numpy as np

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
    '-d',
    type=Path,
    help='path to data directory',
)
parser.add_argument(
    '--weights',
    '-w',
    type=Path,
    help='Path to VGG16 weights',
)
parser.add_argument(
    '--save',
    '-s',
    type=Path,
    help='Path to save folder',
)
parser.add_argument(
    '--epochs',
    '-e',
    type=int,
    help='# of epochs to run',
)
parser.add_argument(
    '--workers',
    type=int,
    help='# of workers in dataloader',
    default=2,
)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


def train(args):
    run_id = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir = args.save / run_id
    save_dir.mkdir()
    batch_size = 8

    # Data
    root = args.data
    transforms = get_transform()
    dataset = Pollene1Dataset(root, transforms)
    print(f'Iterations in dataset {len(dataset)//batch_size}')
    data_loader = data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=True,
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

    loss_hist = []

    t0 = time.time()
    for i in range(args.epochs):
        print(f'===== Epoch {i:2d} =====')
        epoch_loss = []
        t1 = time.time()
        for bidx, batch in enumerate(data_loader):
            images, targets, labels = batch.data()
            target_boxes = []
            target_labels = []
            for truth, label in zip(targets, labels):
                target_box, target_label = encode(ssd_net.priors, truth, label)
                target_boxes.append(target_box)
                target_labels.append(target_label)
            gloc = torch.stack(target_boxes, dim=0)
            glabel = torch.stack(target_labels, dim=0).long()

            if args.cuda:
                gloc, glabel, images = gloc.cuda(), glabel.cuda(), images.cuda()

            ploc, pconf = ssd_net(images)
            optimizer.zero_grad()

            loss = criterion(ploc, pconf, gloc, glabel)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            if bidx % 10 == 0:
                print(f'Loss Iter. {bidx:02d}: {loss.item():6.3f}')

        elapsed = int(time.time() - t1)
        loss_hist.extend(epoch_loss)
        torch.save(ssd_net.state_dict(), save_dir / f'ssd_epoch_{i:02d}.pth')

        print(f"Mean Loss:\t{np.mean(epoch_loss):4.2f}")
        print(f"Time:\t{datetime.timedelta(seconds=elapsed)}")

    elapsed = int(time.time() - t0)
    loss_file = save_dir / 'loss_hist.pkl'
    pickle.dump(loss_hist, loss_file.open('wb'))
    print('====== FINISH ======')
    print(f"Time:\t{datetime.timedelta(seconds=elapsed)}")


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
