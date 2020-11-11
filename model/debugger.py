import torch

from ssd import make_ssd
from utils import decode

if __name__ == "__main__":
    m = make_ssd()
    x = torch.randn(1, 3, 300, 300)
    l, c = m(x)
    print(decode(m.priors, l.abs(), c, soft=False))
