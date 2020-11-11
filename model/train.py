from random import randint
import time

import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data
from torch.optim import SGD
from torchvision import datasets

from ssd import make_ssd
from loss import MultiBoxLoss
from utils import encode


def collate(batch):
    images = []
    targets = []
    labels = []
    to_tensor = transforms.ToTensor()
    for im, t in batch:
        num_targets = randint(1, 3)
        t_min_xy = torch.randint(130, 140, (num_targets, 2))
        t_max_xy = torch.randint(160, 180, (num_targets, 2))
        targets.append(torch.cat((t_min_xy, t_max_xy), dim=1))
        labels.append(torch.ones(num_targets))
        images.append(to_tensor(im))
    return torch.stack(images, dim=0), targets, labels


def train():
    batch_size = 8

    dataset = datasets.FakeData(1000, (3, 300, 300), 2)
    ssd_net = make_ssd(300, 2)
    optimizer = SGD(ssd_net.parameters(), lr=1e-3, momentum=1e-5)
    criterion = MultiBoxLoss(ssd_net.priors, batch_size)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate)

    ssd_net.train()

    batch_iterator = iter(data_loader)

    epochs = 3
    for i in range(epochs):
        images, targets, labels = next(batch_iterator)
        target_boxes = []
        target_labels = []
        for truth, label in zip(targets, labels):
            target_box, target_label = encode(ssd_net.priors, truth, label)
            target_boxes.append(target_box)
            target_labels.append(target_label)
        gloc = torch.stack(target_boxes, dim=0)
        glabel = torch.stack(target_labels, dim=0).long()

        t0 = time.time()
        ploc, pconf = ssd_net(images)
        t1 = time.time()
        optimizer.zero_grad()

        loss = criterion(ploc, pconf, gloc, glabel)
        loss.backward()
        optimizer.step()
        t2 = time.time()
        print(f'===== Iteration {i:2d} =====')
        print(f"Loss:\t{float(loss):3.1f}")
        print(f"Forward:\t{t1-t0:3.1f} sec")
        print(f"Backward:\t{t2-t1:3.1f} sec")
        print(f"Total:\t\t{t2-t0:3.1f} sec")


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.FloatTensor')
    train()
