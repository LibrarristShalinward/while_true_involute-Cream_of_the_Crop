'''
本文件为timm.models.layers.conv2d_same的全复制修改
由于引用过于复杂，故不对未被调用的函数进行筛除
'''

import paddle
from typing import Tuple, Optional
from .padding import pad_same, get_padding_value
from paddle.nn import Conv2D
import paddle.nn.functional as F


def conv2d_same(
        x, weight: paddle.Tensor, bias: Optional[paddle.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(Conv2D):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias_attr = None):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias_attr = bias_attr)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias_attr', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return Conv2D(in_chs, out_chs, kernel_size, padding=padding, **kwargs)