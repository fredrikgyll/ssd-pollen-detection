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
from model.ssd import SSD, ResNet, make_ssd
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
parser.add_argument(
    '--checkpoint', '-cp', type=Path, help='Path to modelstate for continued training'
)
parser.add_argument('--save', '-s', type=Path, help='Path to save folder')
parser.add_argument('--iter', '-i', type=int, help='# of Iterations')
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
    default=[10000],
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
    from model.utils.bunny import CDNConnector

    conn = CDNConnector(api_key, 'pollen')
    conn.upload_file('models/', pth, file_name=run_id + '.pth')


def save_name(root: Path, iteration):
    return root / f'ssd_{iteration:06d}.pth'


def save_state(model, optimizer, criterion, scheduler, iteration, dir):
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'criterion': criterion,
            'iteration': iteration,
        },
        save_name(dir, iteration),
    )


def load_state(pth, model, optimizer, scheduler):
    checkpoint = torch.load(pth, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    iteration = checkpoint['iteration']
    criterion = checkpoint['criterion']
    return iteration, criterion


def train(
    model,
    data_loader,
    optimizer,
    criterion,
    scheduler,
    max_iter,
    save_dir,
    logger,
    start_iter=1,
    live=False,
    cuda=True,
):
    table_widths = [10, 9, 9, 9, 9, 9]
    log_row = '| {{:>{}}} | {{:{}.4f}} | {{:{}.4f}} | {{:{}.4f}} | {{:>{}}} | {{:>{}}} |'
    log_row = log_row.format(*table_widths)
    row_width = 4 + sum(table_widths) + 3 * (len(table_widths) - 1)
    header = ['Iteration', 'Loc loss', 'Conf Loss', 'Loss Sum', 'Elapsed', 'ETA']
    logger('Training log'.center(row_width))
    logger('-' * row_width)
    header_fmt = ' | '.join(
        f'{{:>{x}}}'.format(h) for x, h in zip(table_widths, header)
    )
    logger('| ' + header_fmt + ' |')

    data_iter = iter(data_loader)
    window_loss = [[], []]
    loss_hist = [[], []]
    t0 = time.time()
    for i in range(start_iter, max_iter + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)
        finally:
            images, targets = batch.data()

        with set_default_tensor_type(torch.cuda.FloatTensor):
            if cuda:
                images = images.cuda()
                targets = [t.cuda() for t in targets]

            out = model(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            scheduler.step()
            window_loss[0].append(loss_l.item())
            window_loss[1].append(loss_c.item())
        if live:
            plotter.plot('loss', 'loc', 'Loss', i, loss_l.item())
            plotter.plot('loss', 'conf', 'Loss', i, loss_c.item())
            plotter.plot('loss', 'total', 'Loss', i, loss.item())
        if i % 50 == 0:
            elapsed = int(time.time() - t0)
            eta = int((elapsed / i) * (max_iter - i))
            loss_l_hist = np.mean(window_loss[0])
            loss_c_hist = np.mean(window_loss[1])
            data_row = [
                i,
                loss_l_hist,
                loss_c_hist,
                loss_l_hist + loss_c_hist,
                f'{datetime.timedelta(seconds=elapsed)}',
                f'{datetime.timedelta(seconds=eta)}',
            ]
            logger(log_row.format(*data_row))
            loss_hist[0].extend(window_loss[0])
            loss_hist[1].extend(window_loss[1])
            window_loss = [[], []]
        if i % 100 == 0:
            save_state(model, optimizer, criterion, scheduler, i, save_dir)

    loss_file: Path = save_dir / 'loss_hist.pkl'
    loss_file.write_bytes(pickle.dumps(loss_hist))

    elapsed = int(time.time() - t0)
    logger('-' * row_width)
    logger(f"Time:\t{datetime.timedelta(seconds=elapsed)}")


if __name__ == "__main__":
    args = parser.parse_args()
    load_dotenv()
    if args.viz:
        print("Live plotting enabled")
        global plotter
        plotter = VisdomLinePlotter(env_name='main')

    run_id = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir = args.save / run_id
    save_dir.mkdir()
    logger = Logger(save_dir)
    logger(f'ID {run_id}')
    logger('Parameters:')
    logger(pformat(vars(args), indent=4))

    # Data
    root = args.data
    batch_size = args.batch_size
    transforms = SSDAugmentation()
    dataset = HDF5Dataset(root, 'balanced1', 'train', transforms)
    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=True,
    )
    num_classes = len(dataset.labels) + 1
    logger(f'Iterations per epoch: {len(dataset)//batch_size}')

    # Model init
    cfg = dict(
        size=300,
        num_classes=num_classes,
        default_boxes=[4, 6, 6, 6, 4, 4],
        variances=[0.1, 0.2],
    )

    model = make_ssd(
        num_classes=num_classes,
        backbone='resnet34',
        backbone_path=args.weights,
        phase='train',
        cfg=cfg,
    )
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=args.multistep,
        gamma=0.1,
    )
    if args.checkpoint:
        start_iter, criterion = load_state(args.checkpoint, optimizer, scheduler)
        if start_iter >= args.iter:
            logger('Max iterations already reached in checkpoint. Ending run.')
            exit(1)
        logger('Starting from checkpoint at iteration: {start_iter}')
    else:
        start_iter = 1
        criterion = MultiBoxLoss([0.1, 0.2], args.cuda)

        model._init_weights()

    logger(f'Number of priors is {model.priors.size(1)}')
    logger(f'Number of extractor layers: {len(model.loc_head)+1}\n')

    model.train()
    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    train(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        max_iter=args.iter,
        save_dir=save_dir,
        logger=logger,
        start_iter=start_iter,
        live=args.viz,
        cuda=args.cuda,
    )

    if args.push:
        logger('Pushing file to bunny...')
        push_file(os.getenv('BUNNY_API_KEY'), run_id, save_name(save_dir, args.iter))
        logger('Done.')
