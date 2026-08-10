# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Coordinate Attention Module for YOLO11
Based on: https://github.com/houqb/CoordAttention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import necessary modules
from .conv import Conv
from .block import Bottleneck
from .block import C2f, C3


class h_sigmoid(nn.Module):
    """Hard sigmoid activation function."""
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """Hard swish activation function."""
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """
    Coordinate Attention Module.
    
    This module captures long-range dependencies with precise positional information
    by encoding channel relationships and spatial information along one spatial direction
    and then decoding the attention weights along the other spatial direction.
    
    Args:
        inp (int): Input channels
        oup (int): Output channels  
        reduction (int): Reduction ratio for intermediate channels
    """
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


# -------- Coordinate Attention (채널 동일, 파라미터 경량) --------
class CoordAtt(nn.Module):
    def __init__(self, channels:int, reduction:int=32):
        super().__init__()
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mip)
        self.act   = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, channels, 1, bias=True)
        self.conv_w = nn.Conv2d(mip, channels, 1, bias=True)
    def forward(self, x):
        b,c,h,w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0,1,3,2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w.permute(0,1,3,2)))
        return x * a_h * a_w

class C2fCoordAtt(C2f):
    def __init__(self, *args, reduction:int=32, **kwargs):
        super().__init__(*args, **kwargs)
        # 임시 디버그
        assert isinstance(self.c2, int), f"self.c2 not int: {type(self.c2)} {self.c2}"
        # YOLOv11 C2f 내부에서 self.c (hidden channels)도 숫자인지 확인
        assert isinstance(getattr(self, 'c', 0), int), f"self.c not int: {type(getattr(self, 'c', None))} {getattr(self, 'c', None)}"
        self.ca = CoordAtt(self.c2, reduction=reduction)

    def forward(self, x):
        return self.ca(super().forward(x))

class C3CoordAtt(C3):
    def __init__(self, *args, reduction:int=32, **kwargs):
        super().__init__(*args, **kwargs)
        c2 = (args[1] if len(args)>=2 and isinstance(args[1], int)
              else kwargs.get('c2', getattr(self, 'c2', None)))
        if not isinstance(c2, int):
            raise ValueError(f"C3CoordAtt: invalid c2 {c2}")
        self.ca = CoordAtt(c2, reduction=reduction)
    def forward(self, x):
        return self.ca(super().forward(x))