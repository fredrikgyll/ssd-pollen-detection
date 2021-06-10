import argparse
import datetime
import os
import pickle
import time
from contextlib import contextmanager
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
import torch.utils.data as data
from dotenv import load_dotenv
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from model.loss import MultiBoxLoss
from model.ssd import SSD, ResNet
from model.utils.augmentations import SSDAugmentation
from model.utils.data import HDF5Dataset, collate
from model.utils.logger import Logger
from model.utils.plotting import VisdomLinePlotter

parser = argparse.ArgumentParser(description='Train SSD300 model')
parser.add_argument(
    '--cuda', action='store_true', help='Train model on cuda enabled GPU'
)
parser.add_argument(
    '--viz', action='store_true', help='Enable live plotting with Visdom'
)
parser.add_argument('--data', '-d', type=Path, help='path to data directory')
parser.add_argument('--weights', '-w', type=Path, help='Path to VGG16 weights')
parser.add_argument('--save', '-s', type=Path, help='Path to save folder')
parser.add_argument('--epochs', '-e', type=int, help='# of epochs to run')
parser.add_argument('--workers', default=2, type=int, help='# of workers in dataloader')
parser.add_argument(
    '--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate'
)
parser.add_argument(
    '--momentum', default=0.9, type=float, help='Momentum value for optim'
)
parser.add_argument(
    '--weight_decay', default=5e-4, type=float, help='Weight decay for SGD'
)
parser.add_argument(
    '--batch-size',
    '--bs',
    type=int,
    default=8,
    help='number of examples for each iteration',
)
parser.add_argument(
    '--multistep',
    nargs='*',
    type=int,
    default=[40, 55],
    help='epochs at which to decay learning rate',
)
parser.add_argument('--push', action='store_true', help='Push final model to bunnyCDN')


@contextmanager
def set_default_tensor_type(tensor_type):
    if torch.tensor(0).is_cuda:
        old_tensor_type = torch.cuda.FloatTensor
    else:
        old_tensor_type = torch.FloatTensor

    torch.set_default_tensor_type(tensor_type)
    yield
    torch.set_default_tensor_type(old_tensor_type)


def push_file(api_key: str, run_id: str, pth: Path):
    from utils.bunny import CDNConnector

    conn = CDNConnector(api_key, 'pollen')
    conn.upload_file('models/', pth, file_name=run_id + '.pth')


def train(args):
    run_id = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir = args.save / run_id
    save_dir.mkdir()
    logger = Logger(save_dir)
    logger(f'ID {run_id}')
    logger('Parameters:')
    logger(pformat(vars(args), indent=4))
    batch_size = args.batch_size

    # Data
    root = args.data
    transforms = SSDAugmentation()
    dataset = HDF5Dataset(root, 'balanced1', 'train', transforms)
    logger(f'Iterations in dataset {len(dataset)//batch_size}')
    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=True,
    )
    num_classes = len(dataset.labels) + 1
    # Model init
    cfg = dict(
        size=300,
        num_classes=num_classes,
        default_boxes=[4, 6, 6, 6, 4, 4],
        variances=[0.1, 0.2],
    )
    ssd_net = SSD(ResNet(backbone='resnet34', backbone_path=args.weights), cfg)
    logger(f'Number of priors is {ssd_net.priors.size(1)}')
    logger(f'Number of extractor layers: {len(ssd_net.loc_head)+1}')
    optimizer = SGD(
        ssd_net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.multistep, gamma=0.1)

    criterion = MultiBoxLoss([0.1, 0.2], args.cuda)

    if args.cuda:
        ssd_net = ssd_net.cuda()
        criterion = criterion.cuda()

    # Weights
    ssd_net._init_weights()

    ssd_net.train()

    loss_hist = []

    t0 = time.time()
    iteration = 0
    for i in range(args.epochs):
        logger(f'===== Epoch {i:2d} =====')
        epoch_loss = []
        t1 = time.time()
        for bidx, batch in enumerate(data_loader):
            images, targets = batch.data()
            with set_default_tensor_type(torch.cuda.FloatTensor):

                if args.cuda:
                    images = images.cuda()
                    targets = [t.cuda() for t in targets]

                out = ssd_net(images)
                # [print(x.size()) for x in out]
                # backprop
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss.append(loss.item())
            if args.viz:
                plotter.plot('loss', 'loc', 'Loss', iteration, loss_l.item())
                plotter.plot('loss', 'conf', 'Loss', iteration, loss_c.item())
                plotter.plot('loss', 'total', 'Loss', iteration, loss.item())
            iteration += 1
            if bidx % 10 == 0:
                logger(f'Loss Iter. {bidx:02d}: {loss.item():6.3f}')

        elapsed = int(time.time() - t1)
        total_elapsed = int(time.time() - t0)
        eta = int((total_elapsed / (i + 1)) * (args.epochs - i - 1))
        loss_hist.extend(epoch_loss)
        if (i + 1) % 10 == 0:
            torch.save(ssd_net.state_dict(), save_dir / f'ssd_epoch_{i:02d}.pth')

        logger(f"Mean Loss:\t{np.mean(epoch_loss):4.2f}")
        logger(f"Time:\t{datetime.timedelta(seconds=elapsed)}")
        logger(f"ETA:\t{datetime.timedelta(seconds=eta)}")
    elapsed = int(time.time() - t0)
    loss_file = save_dir / 'loss_hist.pkl'
    last_model_path = save_dir / 'ssd_last.pth'
    torch.save(ssd_net.state_dict(), last_model_path)
    pickle.dump(loss_hist, loss_file.open('wb'))
    logger('====== FINISH ======')
    logger(f"Time:\t{datetime.timedelta(seconds=elapsed)}")
    if args.push:
        logger('Pushing file to bunny...')
        push_file(os.getenv('BUNNY_API_KEY'), run_id, last_model_path)
        logger('Done.')


if __name__ == "__main__":
    args = parser.parse_args()
    load_dotenv()
    if args.viz:
        print("plotting")
        global plotter
        plotter = VisdomLinePlotter(env_name='main')
    train(args)
