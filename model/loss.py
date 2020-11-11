import torch
import torch.nn as nn

from torch import Tensor
from utils import point_form


class MultiBoxLoss(nn.Module):
    def __init__(self, dboxes: Tensor, batch_size: int) -> None:
        super(MultiBoxLoss, self).__init__()
        self.dboxes = nn.Parameter(
            point_form(dboxes).expand(batch_size, -1, -1), requires_grad=False
        )
        self.batch_size = batch_size
        self.loc_loss_func = nn.SmoothL1Loss(reduction='none')
        self.conf_loss_func = nn.CrossEntropyLoss(reduction='none')

    def _to_offsets(self, gloc):
        assert gloc.size() == torch.Size([self.batch_size, 8732, 4])

        dxy = self.dboxes[:, :, :2]
        dwh = self.dboxes[:, :, 2:]
        gxy = gloc[:, :, :2]
        gwh = gloc[:, :, 2:]
        gxy = (gxy - dxy) / dwh
        gwh = (gwh / dwh).log()
        return torch.cat((gxy, gwh), dim=2).contiguous()

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
        self, ploc: Tensor, pconf: Tensor, gloc: Tensor, glabel: Tensor
    ) -> Tensor:
        ploc = ploc.transpose(1, 2)
        assert ploc.size() == torch.Size([self.batch_size, 8732, 4])
        assert gloc.size() == torch.Size([self.batch_size, 8732, 4])
        assert glabel.size() == torch.Size([self.batch_size, 8732])
        assert pconf.size() == torch.Size([self.batch_size, 2, 8732])
        assert ploc.size() == gloc.size()

        mask = glabel > 0
        num_pos = mask.sum(dim=1)

        gloc_offset = self._to_offsets(gloc)
        loc_loss = self.loc_loss_func(ploc, gloc_offset)
        loc_loss = loc_loss.sum(dim=2)
        loc_loss = (mask * loc_loss).sum(dim=1)

        closs = self.conf_loss_func(pconf, glabel)
        con_loss = self.hard_negative_mining(closs, mask, num_pos)

        # avoid no object detected
        total_loss = loc_loss + con_loss
        num_mask = (num_pos > 0).float()
        num_pos = num_pos.float().clamp(min=1e-6)
        return (total_loss * num_mask / num_pos).mean(dim=0)
