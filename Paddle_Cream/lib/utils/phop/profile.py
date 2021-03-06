import paddle
from paddle import nn
from collections import Iterable

from .basic_hooks import (count_adap_avgpool, count_avgpool, count_bn,
                          count_convNd, count_linear, count_parameters,
                          count_relu, count_upsample, zero_ops)

# 原thop.profile参数
register_hooks = {
    nn.Pad2D: zero_ops,

    nn.Conv1D: count_convNd,
    nn.Conv2D: count_convNd,
    nn.Conv3D: count_convNd,
    nn.Conv1DTranspose: count_convNd,
    nn.Conv2DTranspose: count_convNd,
    nn.Conv3DTranspose: count_convNd,

    nn.BatchNorm1D: count_bn,
    nn.BatchNorm2D: count_bn,
    nn.BatchNorm3D: count_bn,
    nn.SyncBatchNorm: count_bn,

    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: count_relu,

    nn.MaxPool1D: zero_ops,
    nn.MaxPool2D: zero_ops,
    nn.MaxPool3D: zero_ops,
    nn.AdaptiveMaxPool1D: zero_ops,
    nn.AdaptiveMaxPool2D: zero_ops,
    nn.AdaptiveMaxPool3D: zero_ops,

    nn.AvgPool1D: count_avgpool,
    nn.AvgPool2D: count_avgpool,
    nn.AvgPool3D: count_avgpool,
    nn.AdaptiveAvgPool1D: count_adap_avgpool,
    nn.AdaptiveAvgPool2D: count_adap_avgpool,
    nn.AdaptiveAvgPool3D: count_adap_avgpool,

    nn.Linear: count_linear,
    nn.Dropout: zero_ops, 

    nn.Upsample: count_upsample,
    nn.UpsamplingBilinear2D: count_upsample,
    nn.UpsamplingNearest2D: count_upsample,
}

# 原thop.profile.profile
def profile(model: nn.Layer, inputs, custom_ops = None, verbose = True):
    handler_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}
    
    def add_hooks(layer: nn.Layer):
        layer.register_buffer('total_ops', paddle.to_tensor([0.]))
        layer.register_buffer('total_params', paddle.to_tensor([0.]))
        
        layer_type = type(layer)
        fn = None

        if layer_type in custom_ops:
            fn = custom_ops[layer_type]
            if layer_type not in types_collection and verbose:
                print("[INFO] Customize rule %s() %s." % (fn.__qualname__, layer_type))
        elif layer_type in register_hooks:
            fn = register_hooks[layer_type]
            if layer_type not in types_collection and verbose:
                print("[INFO] Register %s() for %s." % (fn.__qualname__, layer_type))
        else:
            if layer_type not in types_collection and verbose:
                prRed("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % layer_type)
        
        if fn is not None:
            handler_collection[layer] = (layer.register_forward_post_hook(fn), layer.register_forward_post_hook(count_parameters))
        types_collection.add(layer_type)
    
    prev_training_status = model.training
    model.eval()
    model.apply(add_hooks)
    with paddle.no_grad():
        model(inputs)
    
    def dfs_count(model: nn.Layer, prefix = "\t"):
        total_ops, total_params = 0, 0
        for layer in model.sublayers():
            if layer in handler_collection and not isinstance(layer, (nn.Sequential, nn.LayerList)):
                m_ops, m_params = layer.total_ops.item(), layer.total_params.item()
            else:
                m_ops, m_params = dfs_count(layer, prefix = prefix + "\t")
            total_ops += m_ops
            total_params += m_params
        return total_ops, total_params
    
    total_ops, total_params = dfs_count(model)

    model.train()
    model.training = prev_training_status
    for layer, (op_handler, params_handler) in handler_collection.items():
        op_handler.remove()
        params_handler.remove()

    return total_ops, total_params

# 原thop.utils.clever_format
def clever_format(nums, format = "%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums, )

    return clever_nums

# 原thop.profile.prRed
def prRed(skk): 
    print("\033[91m{}\033[00m".format(skk))
