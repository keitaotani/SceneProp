from torch import nn

from .swin_transformer import SwinTransformer
from .fpn import FPN


class SwinFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = SwinTransformer()
        self.fpn = FPN()

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x