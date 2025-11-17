from typing import Tuple
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class PositionalEncoding2D(nn.Module):
    def __init__(self, meshsize, featdim, lowfreq_thresh):
        super().__init__()
        assert featdim % 2 == 0
        featdim = featdim // 2

        self.meshsize = meshsize

        upper = meshsize // 2 + 1
        lower = upper - meshsize
        xi, eta = torch.meshgrid(torch.arange(lower, upper), torch.arange(upper), indexing='ij')
        xieta = torch.stack([xi.flatten(), eta.flatten()], dim=1)
        freq = torch.norm(xieta.float(), dim=1)
        mask_low = freq <= lowfreq_thresh
        mask_high = (freq > lowfreq_thresh) & (freq <= meshsize)
        xieta_low = xieta[mask_low]
        xieta_high = xieta[mask_high]
        high_idx = torch.randperm(len(xieta_high))[:featdim - len(xieta_low)]
        xieta_high = xieta_high[high_idx]
        xieta = torch.cat([xieta_low, xieta_high], dim=0)
        thetaphi = 2 * torch.pi * xieta / meshsize
        
        self.register_buffer('theta', thetaphi[:, 0])
        self.register_buffer('phi', thetaphi[:, 1])
    
    def forward(self, x, y, sigma_x, sigma_y):
        x = x[..., None] * self.meshsize
        y = y[..., None] * self.meshsize
        sigma_x = sigma_x[..., None] * self.meshsize
        sigma_y = sigma_y[..., None] * self.meshsize

        base = torch.exp(- sigma_x ** 2 * (1 - torch.cos(self.theta)) - sigma_y ** 2 * (1 - torch.cos(self.phi)))
        phase_shift = self.theta * x + self.phi * y
        cosenc = base * torch.cos(phase_shift)
        sinenc = base * torch.sin(phase_shift)
        enc = torch.cat([cosenc, sinenc], dim=-1)
        return enc

    def to_image_demo(self, x: float, y: float, sigma_x: float, sigma_y: float):
        fourier = torch.zeros(self.meshsize, self.meshsize // 2 + 1, dtype=torch.complex64)

        xi = torch.round(self.theta * self.meshsize / 2 / np.pi).long()
        eta = torch.round(self.phi * self.meshsize / 2 / np.pi).long()
        enc = self(torch.tensor(x), torch.tensor(y), torch.tensor(sigma_x), torch.tensor(sigma_y))
        cosenc = enc[..., :enc.shape[-1] // 2]
        sinenc = enc[..., enc.shape[-1] // 2:]

        fourier[xi, eta] = cosenc - 1j * sinenc

        return torch.fft.irfft2(fourier, s=(self.meshsize, self.meshsize)).T


class TransformerPositionalEncoding2D(nn.Module):

    def __init__(self, featdim):
        super().__init__()
        assert featdim % 8 == 0
        self.featdim = featdim

        print(f"Using TransformerPositionalEncoding2D with featdim={featdim}")

    def forward(self, x, y, sigma_x, sigma_y):

        def pe1d(x, d):
            device = x.device
            div_term = torch.exp(torch.arange(0, d, 2, device=device) * (-math.log(10000.0) / d))
            x = x.unsqueeze(-1) * 500
            pe = torch.cat([torch.sin(x * div_term), torch.cos(x * div_term)], dim=-1)
            return pe

        x1 = x - sigma_x / 2
        x2 = x + sigma_x / 2
        y1 = y - sigma_y / 2
        y2 = y + sigma_y / 2

        enc = torch.cat([
            pe1d(x1, self.featdim // 4),
            pe1d(x2, self.featdim // 4),
            pe1d(y1, self.featdim // 4),
            pe1d(y2, self.featdim // 4)
        ], dim=-1)

        return enc


class NoPositionalEncoding(nn.Module):
    def __init__(self, featdim):
        super().__init__()
        self.featdim = featdim
    
    def forward(self, x, y, sigma_x, sigma_y):
        shape = (*x.shape, self.featdim)
        return torch.zeros(shape, device=x.device, dtype=torch.float32)


class RegressBoxesPositionalEncoding(nn.Module):
    def __init__(self, meshsize, featdim, lowfreq_thresh, enctype='random_fourier'):
        super().__init__()
        if enctype == 'random_fourier':
            self.encoder = PositionalEncoding2D(meshsize, featdim, lowfreq_thresh)
        elif enctype == 'transformer':
            self.encoder = TransformerPositionalEncoding2D(featdim)
        elif enctype == 'none':
            self.encoder = NoPositionalEncoding(featdim)
        else:
            raise ValueError(f"Unknown encoding type: {enctype}")
    
    def forward(self, bbox_reg: torch.Tensor, relative_anchor_size_to_pixel: Tuple[int, int]):
        _, _, h, w = bbox_reg.shape
        max_size = max(h, w)
        x = torch.arange(w, dtype=torch.float32, device=bbox_reg.device) / max_size + 1 / (2 * max_size)
        y = torch.arange(h, dtype=torch.float32, device=bbox_reg.device) / max_size + 1 / (2 * max_size)
        x, y = torch.meshgrid(x, y, indexing='xy')

        wx, wy, ww, wh = (10., 10., 5., 5.)
        dx = bbox_reg[:, 0] / wx  # (batch, height, width)
        dy = bbox_reg[:, 1] / wy
        dw = bbox_reg[:, 2] / ww
        dh = bbox_reg[:, 3] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=math.log(1000. / 16))
        dh = torch.clamp(dh, max=math.log(1000. / 16))

        anchor_w = relative_anchor_size_to_pixel[0] / max_size
        anchor_h = relative_anchor_size_to_pixel[1] / max_size

        pred_ctr_x = dx * anchor_w + x
        pred_ctr_y = dy * anchor_h + y
        pred_w = torch.exp(dw) * anchor_w
        pred_h = torch.exp(dh) * anchor_h

        encoding = self.encoder(pred_ctr_x, pred_ctr_y, pred_w / 2, pred_h / 2)
        encoding = encoding.permute(0, 3, 1, 2)  # (batch, featdim, height, width)

        return encoding


class RelationshipProjection(nn.Module):
    def __init__(self, in_vis_dim:int, in_pos_dim:int, out_relationship_dim:int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_vis_dim + in_pos_dim, in_vis_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_vis_dim, in_vis_dim, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.shift_proj = nn.Conv2d(in_vis_dim, in_pos_dim, kernel_size=1)
        self.pos_proj = nn.Conv2d(in_pos_dim, out_relationship_dim, kernel_size=1)
        self.vis_proj = nn.Conv2d(in_vis_dim, out_relationship_dim, kernel_size=1)
        self.out_relationship_dim = out_relationship_dim
    
    def forward(self, visual_features, positional_features):
        feature = self.feature(torch.cat([visual_features, positional_features], dim=1))
        shift = self.shift_proj(feature)
        pos = self.pos_proj(positional_features * shift)
        vis = self.vis_proj(feature)
        vec = vis + pos
        vec = vec / (vec.norm(p=2, dim=1, keepdim=True) * self.out_relationship_dim) ** 0.5
        return vec


class RPN(nn.Module):
    def __init__(self, channels:int = 256, anchor_sizes = (64, 128, 256, 512, 1024), pe_type='random_fourier'):
        super().__init__()

        self.anchor_sizes = anchor_sizes

        num_anchors = 1

        # For box regression
        self.bbox_pred = nn.Conv2d(channels, num_anchors * 4, kernel_size=1)
        self.objectness_for_nms = nn.Conv2d(channels, num_anchors * 1, kernel_size=1)

        for conv in [self.bbox_pred, self.objectness_for_nms]:
            nn.init.normal_(conv.weight, std=0.01)
            nn.init.constant_(conv.bias, 0)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])
        self.positional_encoding = RegressBoxesPositionalEncoding(meshsize=96, featdim=channels, lowfreq_thresh=6, enctype=pe_type)

    def forward(self, visual_features):
        # For box regression
        bbox_reg = []
        objectness_for_nms = []
        features = []

        for visual, scale in zip(visual_features, self.scales):
            reg = scale(self.bbox_pred(visual))
            bbox_reg.append(reg)
            objectness_for_nms.append(self.objectness_for_nms(visual))
            
            posenc = self.positional_encoding(reg, (8, 8))
            features.append(torch.cat([visual, posenc], dim=1))

        return (
            bbox_reg,       # For box regression
            objectness_for_nms, # For NMS
            features        # number of channel = 2 * channels
        )