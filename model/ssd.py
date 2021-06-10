from typing import Tuple, Dict, Any, List
from numpy.lib.utils import source

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

from priors import PriorBox
from utils.detection import Detect

Modules = List[nn.Module]


class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', backbone_path=None):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=False)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=False)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=False)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=False)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=False)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD(nn.Module):
    """
    Implementation of the Single Shot MultiBox Detector Architecture

    base:
        Standard ResNet
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

    def __init__(self, base: ResNet, cfg, phase: str = 'train'):
        super(SSD, self).__init__()

        self.num_classes = cfg['num_classes']
        self.size = cfg['size']
        self.default_boxes = cfg['default_boxes']
        priors = PriorBox()
        self.priors = nn.Parameter(priors.forward(), requires_grad=False)
        self.phase = phase

        self.base: ResNet = base
        self.extra = self._extra_layers(base.out_channels)

        _loc_layers, _conf_layers = [], []
        for oc, nd in zip(self.base.out_channels, self.default_boxes):
            _loc_layers.append(nn.Conv2d(oc, 4 * nd, kernel_size=3, padding=1))
            _conf_layers.append(
                nn.Conv2d(oc, self.num_classes * nd, kernel_size=3, padding=1)
            )
        self.loc_head = nn.ModuleList(_loc_layers)
        self.conf_head = nn.ModuleList(_conf_layers)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(
                num_classes=self.num_classes, 
                bkg_label=0, 
                top_k=50, 
                area_thresh=0.1,
                conf_thresh=0.1, 
                nms_thresh=0.45, 
                variances=cfg['variances']
            )

    def _init_weights(self):
        layers = [*self.extra, *self.loc_head, *self.conf_head]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.base(x)

        source_layers = [x]
        for layer in self.extra:
            x = layer(x)
            source_layers.append(x)

        loc: List[Tensor] = []
        conf: List[Tensor] = []
        for x, l, c in zip(source_layers, self.loc_head, self.conf_head):
            # dims: batch, row, col, class/offset per aspect
            loc.append(l(x).view(x.size(0), 4, -1))
            conf.append(c(x).view(x.size(0), self.num_classes, -1))

        # dims: batch, offsets/class scale-row-col-aspect
        loc_tensor: Tensor = torch.cat(loc, 2).contiguous()
        conf_tensor: Tensor = torch.cat(conf, 2).contiguous()

        if self.phase == "train":
            output = (
                loc_tensor,
                conf_tensor,
                self.priors,
            )
        else:
            output = self.detect(
                loc_tensor.transpose(1, 2),  # loc preds
                F.softmax(conf_tensor.transpose(1, 2), dim=-1),
                self.priors.transpose(0, 1).type_as(x),  # default boxes
            )
        return output

    def _extra_layers(self, in_channels: List[int]) -> nn.ModuleList:
        # Extra layers for feature scaling
        channels = [256, 256, 128, 128, 128]
        layers = []
        for i, (ins, outs, middles) in enumerate(
            zip(in_channels[:-1], in_channels[1:], channels)
        ):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(ins, middles, kernel_size=1, bias=False),
                    nn.BatchNorm2d(middles),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        middles, outs, kernel_size=3, padding=1, stride=2, bias=False
                    ),
                    nn.BatchNorm2d(outs),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(ins, middles, kernel_size=1, bias=False),
                    nn.BatchNorm2d(middles),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(middles, outs, kernel_size=3, bias=False),
                    nn.BatchNorm2d(outs),
                    nn.ReLU(inplace=True),
                )
            layers.append(layer)
        return nn.ModuleList(layers)


def make_ssd(
    num_classes: int = 2, backbone: str = 'resnet34', phase: str = 'train'
) -> SSD:
    cfg = dict(
        size=300,
        num_classes=num_classes,
        default_boxes=[4, 6, 6, 6, 4, 4],
        variances=[0.1, 0.2],
    )
    base = ResNet(backbone=backbone)
    return SSD(base, cfg, phase=phase)
