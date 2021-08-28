'''
本文件为timm.models.layers.cond_conv2d的全复制修改
由于引用过于复杂，故不对未被调用的函数进行筛除
'''

import math
from functools import partial
import numpy as np
from paddle.framework import dtype

from paddle.nn import Layer, initializer, functional
from paddle import create_parameter, matmul#, ones

from .helpers import tup_pair
from .padding import get_padding_value
from .conv2d_same import conv2d_same

def get_condconv_initializer(initializer, num_experts, expert_shape):
    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if (len(weight.shape) != 2 or weight.shape[0] != num_experts or
                weight.shape[1] != num_params):
            raise (ValueError(
                'CondConv variables must have shape [num_experts, num_params]'))
        for i in range(num_experts):
            initializer(weight[i].reshape(expert_shape))
    return condconv_initializer


class CondConv2D(Layer):
    __constants__ = ['bias', 'in_channels', 'out_channels', 'dynamic_padding']

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, groups=1, bias_attr=False, num_experts=4):
        super(CondConv2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tup_pair(kernel_size)
        self.stride = tup_pair(stride)
        padding_val, is_padding_dynamic = get_padding_value(
            padding, kernel_size, stride=stride, dilation=dilation)
        self.dynamic_padding = is_padding_dynamic
        self.padding = tup_pair(padding_val)
        self.dilation = tup_pair(dilation)
        self.groups = groups
        self.num_experts = num_experts

        self.weight_shape = (self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = create_parameter(
            shape = (self.num_experts, weight_num_param), 
            dtype = "float32")

        assert bias_attr in [None, False]
        if bias_attr == False:
            self.bias_shape = (self.out_channels,)
            self.bias = create_parameter(
                shape = (self.num_experts, self.out_channels), 
                dtype = "float32"
            )
        else:
            self.add_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init_weight = get_condconv_initializer(
            # partial(initializer.KaimingUniform, a=math.sqrt(5)), self.num_experts, self.weight_shape)
            initializer.KaimingUniform(), self.num_experts, self.weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(
                initializer.Uniform(low = -bound, high = bound), self.num_experts, self.bias_shape)
            init_bias(self.bias)

    def forward(self, x, routing_weights):
        B, C, H, W = x.shape
        # routing_weights = ones((self.weight.shape[0], 1), dtype = "float32") if routing_weights == None else routing_weights
        weight = matmul(routing_weights, self.weight)
        new_weight_shape = (B * self.out_channels, self.in_channels // self.groups) + self.kernel_size
        weight = weight.reshape(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = matmul(routing_weights, self.bias)
            bias = bias.reshape(B * self.out_channels)
        x = x.reshape(1, B * C, H, W)
        if self.dynamic_padding:
            out = conv2d_same(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * B)
        else:
            out = functional.conv2d(
                x, weight, bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * B)
        out = out.permute([1, 0, 2, 3]).reshape(B, self.out_channels, out.shape[-2], out.shape[-1])
        return out