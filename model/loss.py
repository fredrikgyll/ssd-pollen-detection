import torch
import torch.nn as nn

from torch import Tensor


class MultiBoxLoss(nn.Module):
    def __init__(self, dboxes: Tensor) -> None:
        super(MultiBoxLoss, self).__init__()
        self.dboxes = dboxes
        self.loc_loss_func = nn.SmoothL1Loss(reduce=False)
        self.conf_loss_func = nn.CrossEntropyLoss(reduce=False)

    def _to_offsets(self, gloc):
        dxy = self.dboxes[:, :2, :]
        dwh = self.dboxes[:, 2:, :]
        gxy = gloc[:, :2, :]
        gwh = gloc[:, 2:, :]
        gxy = (gxy - dxy) / dwh
        gwh = (gwh / dwh).log()
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
        closs = (loss * (mask.float() + neg_mask.float())).sum(dim=1)
        return closs

    def forward(
        self, ploc: Tensor, plabel: Tensor, gloc: Tensor, glabel: Tensor
    ) -> Tensor:
        mask = glabel > 0
        num_pos = mask.sum(dim=1)

        gloc_offset = self._to_offsets(gloc)
        loc_loss = self.loc_loss_func(ploc, gloc_offset).sum(dim=1)
        loc_loss = (mask * loc_loss).sum(dim=1)

        closs = self.conf_loss_func(plabel, glabel)
        con_loss = self.hard_negative_mining(closs, mask, num_pos)

        # avoid no object detected
        total_loss = loc_loss + con_loss
        num_mask = (num_pos > 0).float()
        num_pos = num_pos.float().clamp(min=1e-6)
        return (total_loss * num_mask / num_pos).mean(dim=0)
