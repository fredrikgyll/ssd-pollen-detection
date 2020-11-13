from typing import Tuple, Dict, Any, List

import torch
from torch import Tensor
import torch.nn as nn

from priors import priors

Modules = List[nn.Module]


class SSD(nn.Module):
    """
    Implementation of the Single Shot MultiBox Detector Architecture

    base:
        Standard VGG16 Conv layer 1 through 5_3,
        added layers conv6 and conv7 from SSD paper.
        Total layers: 30
        Conv4_3: layer 21
        Conv5_3: layer 28
    extra:
        SSD layers conv8_1 through conv11_2, no layers between.
        Total layers: 8
    head:
        tuple of detection heads. (loc_head, conf_head)
    sizes:
        The dimentions and number of aspect ratios of the 6 feature layers
        layer 0:    38, 4
        layer 1:    19, 6
        layer 2:    10, 6
        layer 3:     5, 6
        layer 4:     3, 4
        layer 5:     1, 4
    """

    def __init__(self, base, extra, head, cfg):
        super(SSD, self).__init__()

        self.num_classes = cfg['num_classes']
        self.size = cfg['size']
        self.source_idx = cfg['source_idx']
        self.default_boxes = cfg['default_boxes']
        self.priors = priors(cfg)
        self.base = nn.ModuleList(base)
        self.extra = nn.ModuleList(extra)
        self.loc_head = nn.ModuleList(head[0])
        self.conf_head = nn.ModuleList(head[1])

    def _init_weights(self):
        layers = [*self.extra, *self.loc_head, *self.conf_head]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        source_layers = []
        # feat_layer = 0
        for idx, l in enumerate(self.base):
            x = l(x)
            if idx in self.source_idx:
                source_layers.append(x)
                # print(f'layer {feat_layer}: {x.size(2)}')
                # feat_layer += 1

        for l in self.extra:
            x = l(x)
            source_layers.append(x)
            # print(f'layer {feat_layer}: {x.size(2)}')
            # feat_layer += 1

        loc: List[Tensor] = []
        conf: List[Tensor] = []
        for x, l, c in zip(source_layers, self.loc_head, self.conf_head):
            # dims: batch, row, col, class/offset per aspect
            # loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            # conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            loc.append(l(x).view(x.size(0), 4, -1))
            conf.append(c(x).view(x.size(0), self.num_classes, -1))
        # dims: batch, offsets/class scale-row-col-aspect
        loc_tensor: Tensor = torch.cat(loc, 2).contiguous()
        conf_tensor: Tensor = torch.cat(conf, 2).contiguous()
        return loc_tensor, conf_tensor


def vgg(cfg: Dict[str, Any], input_channels: int) -> Modules:
    layers: Modules = []
    in_channels = input_channels
    for key in cfg['vgg']:
        if isinstance(key, str):
            if key == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif key == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            layers += [
                nn.Conv2d(in_channels, out_channels=key, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_channels = key
    layers += [
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),  # conv6
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),  # conv7
        nn.ReLU(inplace=True),
    ]
    return layers


def extra_layers(cfg: Dict[str, Any]) -> Modules:
    # Extra layers for feature scaling
    # [*(256, 'S', 512), *(128, 'S', 256), *(128, 256), *(128, 256)],
    channels = [256, 128, 128, 128]
    in_channels = cfg['out_channels'][1:]
    layers = []
    for i, (input_size, output_size, channels) in enumerate(
        zip(in_channels[:-1], in_channels[1:], channels)
    ):
        if i < 2:
            layer = nn.Sequential(
                nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    channels,
                    output_size,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    bias=False,
                ),
                nn.BatchNorm2d(output_size),
                nn.ReLU(inplace=True),
            )
        else:
            layer = nn.Sequential(
                nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                nn.BatchNorm2d(output_size),
                nn.ReLU(inplace=True),
            )
        layers.append(layer)
    return nn.ModuleList(layers)


def multibox_layers(cfg: Dict[str, Any]) -> Modules:
    _loc_layers = []
    _conf_layers = []
    nc = cfg['num_classes']
    for oc, nd in zip(cfg['out_channels'], cfg['default_boxes']):
        _loc_layers.append(nn.Conv2d(oc, 4 * nd, kernel_size=3, padding=1))
        _conf_layers.append(nn.Conv2d(oc, nc * nd, kernel_size=3, padding=1))
    return _loc_layers, _conf_layers


def make_ssd(size: int = 300, num_classes: int = 2) -> SSD:
    cfg = {
        'vgg': [
            *(64, 64, 'M'),  # conv1
            *(128, 128, 'M'),  # conv2
            *(256, 256, 256, 'C'),  # conv3
            *(512, 512, 512, 'M'),  # conv4
            *(512, 512, 512),  # conv5
        ],
        'num_classes': num_classes,
        'size': size,
    }
    if size == 300:
        cfg.update(
            out_channels=[512, 1024, 512, 256, 256, 256],
            source_idx=[21, 33],
            feature_maps=[38, 19, 10, 5, 3, 1],
            default_boxes=[4, 6, 6, 6, 4, 4],
            s_min=0.2,
            s_max=0.9,
            aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        )
    base = vgg(cfg, input_channels=3)
    extra = extra_layers(cfg)
    head = multibox_layers(cfg)
    return SSD(base, extra, head, cfg)
