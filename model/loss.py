from typing import List

import torch
import torch.nn as nn

from model.utils.geometry import encode


class MultiBoxLoss(nn.Module):
    def __init__(self, variances: List, cuda: bool) -> None:
        super(MultiBoxLoss, self).__init__()
        self.loc_loss_func = nn.SmoothL1Loss(reduction='none')
        self.conf_loss_func = nn.CrossEntropyLoss(reduction='none')
        self.variances = variances
        self.use_cuda = cuda

    def _to_offsets(self, gloc, priors, variances):
        # assert gloc.size() == torch.Size([self.batch_size, 4, 8732])

        dxy = priors[:, :2, :]
        dwh = priors[:, 2:, :]
        gxy = gloc[:, :2, :]
        gwh = gloc[:, 2:, :]
        gxy -= dxy
        gxy /= variances[0] * dwh
        gwh = (gwh / dwh).log() / variances[1]
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def hard_negative_mining(self, loss, mask, pos_num):
        # postive mask will never selected
        con_neg = loss.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        # print(con.shape, mask.shape, neg_mask.shape)
        closs = (loss * (mask.float() + neg_mask.float())).sum()
        return closs

    def forward(self, predictions, targets) -> torch.Tensor:
        # def forward(self, predictions, targets):
        # assert ploc.size() == torch.Size([self.batch_size, 4, 8732])
        # assert gloc.size() == torch.Size([self.batch_size, 4, 8732])
        # assert glabel.size() == torch.Size([self.batch_size, 8732])
        # assert pconf.size() == torch.Size([self.batch_size, 2, 8732])
        # assert ploc.size() == gloc.size()

        ploc, pconf, priors = predictions

        # matching
        target_boxes = []
        target_labels = []
        for target in targets:
            target_box, target_label = encode(
                priors.transpose(0, 1), target[:, :4], target[:, 4]
            )
            target_boxes.append(target_box)
            target_labels.append(target_label)
        gloc = torch.stack(target_boxes, dim=0)
        glabel = torch.stack(target_labels, dim=0).long()

        mask = glabel > 0
        num_pos = mask.sum(dim=1)

        gloc_offset = self._to_offsets(gloc, priors.unsqueeze(0), self.variances)
        loc_loss = self.loc_loss_func(ploc, gloc_offset)
        loc_loss = loc_loss.sum(dim=1)
        loc_loss = mask.float() * loc_loss
        loc_loss = torch.sum(loc_loss)

        closs = self.conf_loss_func(pconf, glabel)
        con_loss = self.hard_negative_mining(closs, mask, num_pos)

        # avoid no object detected

        num_pos = num_pos.sum().float()
        loc_loss /= num_pos
        con_loss /= num_pos
        return loc_loss, con_loss
