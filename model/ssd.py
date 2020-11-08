from typing import Tuple, Dict, Any, Union, List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from priors import priors

Modules = List[nn.Module]
LayerSpec = List[Union[str, int]]


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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        source_layers = []
        feat_layer = 0
        for layers, layer_points in zip((self.base, self.extra), self.source_idx):
            for idx, layer in enumerate(layers):
                x = layer(x)
                if idx in layer_points:
                    source_layers.append(x)
                    print(f'layer {feat_layer}: {x.size(2)}')
                    feat_layer += 1

        loc: List[Tensor] = []
        conf: List[Tensor] = []
        for x, l, c in zip(source_layers, self.loc_head, self.conf_head):
            # dims: batch, row, col, class/offset per aspect
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        # dims: batch, offsets/class scale-row-col-aspect
        loc_tensor: Tensor = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf_tensor: Tensor = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        return (
            loc_tensor.view(loc_tensor.size(0), -1, 4),
            conf_tensor.view(conf_tensor.size(0), -1, self.num_classes),
            self.priors,
        )


def vgg(cfg: LayerSpec, input_channels: int) -> Modules:
    layers: Modules = []
    in_channels = input_channels
    for key in cfg:
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
        nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),  # conv6
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),  # conv7
        nn.ReLU(inplace=True),
    ]
    return layers


def extra_layers(cfg: LayerSpec, input_channels: int) -> Modules:
    # Extra layers for feature scaling
    layers: Modules = []
    in_channels = input_channels
    kernel = 1
    i = 0
    layer_cfg: Dict[str, Any] = {}
    while i < len(cfg):
        if cfg[i] == 'S':
            i += 1
            layer_cfg.update(stride=2, padding=1)
        else:
            layer_cfg.update(stride=1, padding=0)
        layer_cfg.update(
            in_channels=in_channels, out_channels=cfg[i], kernel_size=kernel
        )
        layers += [nn.Conv2d(**layer_cfg)]

        kernel = 4 - kernel
        in_channels = cfg[i]
        i += 1

    return layers


def multibox_layers(
    cfg: Dict[str, Any], base: Modules, extra_layers: Modules
) -> Modules:
    _loc_layers = []
    _conf_layers = []

    for layers, layer_points, boxes in zip(
        (base, extra_layers), cfg['source_idx'], cfg['default_boxes']
    ):
        for idx, mbox in zip(layer_points, boxes):
            _loc_layers.append(
                nn.Conv2d(layers[idx].out_channels, 4 * mbox, kernel_size=3, padding=1)
            )
            _conf_layers.append(
                nn.Conv2d(
                    layers[idx].out_channels,
                    cfg['num_classes'] * mbox,
                    kernel_size=3,
                    padding=1,
                )
            )
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
            extra=[*(256, 'S', 512), *(128, 'S', 256), *(128, 256), *(128, 256)],
            source_idx=[[21, 32], [1, 3, 5, 7]],
            feature_maps=[38, 19, 10, 5, 3, 1],
            default_boxes=[
                [4, 6],
                [6, 6, 4, 4],
            ],
            s_min=0.2,
            s_max=0.9,
            aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        )
    elif size == 512:
        cfg.update(
            extra=[
                *(256, 'S', 512),
                *(128, 'S', 256),
                *(128, 'S', 256),
                *(128, 256),
                *(128, 'S', 256),
            ],
            source_idx=[[21, 32], [1, 3, 5, 7, 9]],
            feature_maps=[64, 32, 16, 8, 4, 2, 1],
            default_boxes=[
                [4, 6],
                [6, 6, 6, 4, 4],
            ],
            s_min=0.15,
            s_max=0.9,
            aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        )

    base = vgg(cfg['vgg'], input_channels=3)
    extra = extra_layers(cfg['extra'], input_channels=1024)
    head = multibox_layers(cfg, base, extra)
    return SSD(base, extra, head, cfg)
