'''
本文件为thop.vision.basic_hooks的全复制修改
由于引用过于复杂，故不对未被调用的函数进行筛除
'''
import logging

import paddle
from paddle.nn.layer.conv import _ConvNd

multiply_adds = 1

def count_parameters(layer, x, y):
    total_params = 0
    for p in layer.parameters():
        total_params += paddle.to_tensor([p.numel()], dtype = "float32")
    layer.total_params[0] = total_params

def zero_ops(layer, x, y):
    layer.total_ops += paddle.to_tensor([int(0)], dtype = "float32")

def count_convNd(layer: _ConvNd, x: tuple, y: paddle.Tensor):
    assert type(x[0]) == paddle.Tensor
    x = x[0]
    kernel_ops = paddle.zeros(layer.weight.shape[2:]).numel()
    bias_ops = 1 if layer.bias is not None else 0

    total_ops = y.size * (layer._in_channels // layer._groups * kernel_ops + bias_ops)

    layer.total_ops += paddle.to_tensor([int(total_ops)], dtype = "float32")

def count_bn(layer, x, y):
    x = x[0]
    nelements = x.numel()
    if not layer.training:
        total_ops = 2 * nelements
    layer.total_ops += paddle.to_tensor([int(total_ops)], dtype = "float32")

def count_relu(layer, x, y):
    x = x[0]
    nelements = x.numel()
    layer.total_ops += paddle.to_tensor([int(nelements)], dtype = "float32")

def count_softmax(layer, x, y):
    x = x[0]
    batch_size, nfeatures = x.shape
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    layer.total_ops += paddle.to_tensor([int(total_ops)], dtype = "float32")

def count_avgpool(layer, x, y):
    kernel_ops = 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements
    layer.total_ops += paddle.to_tensor([int(total_ops)], dtype = "float32")

def count_adap_avgpool(layer, x, y):
    kernel = paddle.to_tensor([*(x[0].shape[2:])], dtype = "float32")
    kernel = kernel// paddle.to_tensor(list((layer.output_size,)), dtype = "float32").squeeze()
    total_add = paddle.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements
    layer.total_ops += paddle.to_tensor([int(total_ops)], dtype = "float32")

def count_upsample(layer, x, y):
    if layer.mode not in ("nearest", "linear", "bilinear", "bicubic",):
        logging.warning("mode %s is not implemented yet, take it a zero op" % layer.mode)
        return zero_ops(layer, x, y)

    if layer.mode == "nearest":
        return zero_ops(layer, x, y)

    x = x[0]
    if layer.mode == "linear":
        total_ops = y.size * 5

    elif layer.mode == "bilinear":
        total_ops = y.size * 11

    elif layer.mode == "bicubic":
        ops_solve_A = 224
        ops_solve_p = 35
        total_ops = y.size * (ops_solve_A + ops_solve_p)

    elif layer.mode == "trilinear":
        total_ops = y.size * (13 * 2 + 5)

    layer.total_ops += paddle.to_tensor([int(total_ops)], dtype = "float32")

def count_linear(layer, x, y):
    total_mul = layer.weight.shape[0]
    num_elements = y.numel()
    total_ops = total_mul * num_elements

    layer.total_ops += paddle.to_tensor([int(total_ops)], dtype = "float32")
