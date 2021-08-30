import paddle
from paddle import nn

from ..paddle_extension import Identity
from .create_conv2d import create_conv2d

# 原timm.models.effcientnet_blocks中的参数
_SE_ARGS_DEFAULT = dict(
    gate_fn = nn.functional.sigmoid,
    act_layer=None,
    reduce_mid=False,
    divisor=1)

# 原timm.models.effcientnet_blocks.ConvBnAct
class ConvBnAct(nn.Layer):
    def __init__(
        self, in_chs, out_chs, 
        kernel_size, stride = 1, dilation = 1, pad_type= "", 
        act_layer = nn.ReLU, norm_layer = nn.BatchNorm2D, 
        norm_kwargs = None):
        super(ConvBnAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = create_conv2d(
            in_chs, out_chs, 
            kernel_size, 
            stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(out_chs, **norm_kwargs)
        self.act1 = act_layer()
    
    def feature_info(self, location):
        if location == 'expansion' or location == 'depthwise':
            info = dict(module='act1', hook_type='forward', num_chs=self.conv.out_channels)
        else:  # location == 'bottleneck'
            info = dict(module='', hook_type='', num_chs=self.conv.out_channels)
        return info
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

# 原timm.models.effcientnet_blocks.DepthwiseSeparableConv
class DepthwiseSeparableConv(nn.Layer):
    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 pw_kernel_size=1, pw_act=False, se_ratio=0., se_kwargs=None,
                 norm_layer=nn.BatchNorm2D, norm_kwargs=None, drop_path_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        norm_kwargs = norm_kwargs or {}
        has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.drop_path_rate = drop_path_rate

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, depthwise=True)
        self.bn1 = norm_layer(in_chs, **norm_kwargs)
        self.act1 = act_layer()
        
        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(in_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = None
        
        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs, **norm_kwargs)
        self.act2 = act_layer() if self.has_pw_act else Identity()
    
    def feature_info(self, location):
        if location == 'expansion':
            # no expansion in this block, use depthwise, before SE
            info = dict(module='act1', hook_type='forward', num_chs=self.conv_pw.in_channels)
        elif location == 'depthwise':  # after SE
            info = dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck'
            info = dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)
        return info
    
    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.se is not None:
            x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += residual
        return x

# 原timm.models.layers.drop.drop_path
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    random_tensor = keep_prob + paddle.rand((x.size()[0], 1, 1, 1), dtype=x.dtype, device=x.device)
    random_tensor = paddle.floor(random_tensor)
    output = x / keep_prob * random_tensor
    return output

# 原timm.models.effcientnet_blocks.InvertedResidual
class InvertedResidual(nn.Layer):
    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_kwargs=None, norm_layer=nn.BatchNorm2D, norm_kwargs=None,
                 conv_kwargs=None, drop_path_rate=0.):
        super(InvertedResidual, self).__init__()

        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer()
        
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            padding=pad_type, depthwise=True, **conv_kwargs)
        self.bn2 = norm_layer(mid_chs, **norm_kwargs)
        self.act2 = act_layer()

        if has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = None
        
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_layer(out_chs, **norm_kwargs)

    def feature_info(self, location):
        if location == 'expansion':
            info = dict(module='act1', hook_type='forward', num_chs=self.conv_pw.in_channels)
        elif location == 'depthwise':  # after SE
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck'
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info
    
    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += residual

        return x

# 原timm.models.effcientnet_blocks.SqueezeExcite

class SqueezeExcite(nn.Layer):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=nn.functional.sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv_reduce = nn.Conv2D(in_chs, reduced_chs, 1)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2D(reduced_chs, in_chs, 1)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

# 原timm.models.layers.effcientnet_blocks.round_channels
def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    if not multiplier:
        return channels
    channels *= multiplier
    return make_divisible(channels, divisor, channel_min)

# 原timm.models.effcientnet_blocks.resolve_se_args
def resolve_se_args(kwargs, in_chs, act_layer=None):
    se_kwargs = kwargs.copy() if kwargs is not None else {}
    # fill in args that aren't specified with the defaults
    for k, v in _SE_ARGS_DEFAULT.items():
        se_kwargs.setdefault(k, v)
    # some models, like MobilNetV3, calculate SE reduction chs from the containing block's mid_ch instead of in_ch
    if not se_kwargs.pop('reduce_mid'):
        se_kwargs['reduced_base_chs'] = in_chs
    # act_layer override, if it remains None, the containing block's act_layer will be used
    if se_kwargs['act_layer'] is None:
        assert act_layer is not None
        se_kwargs['act_layer'] = act_layer
    return se_kwargs

# 原timm.models.effcientnet_blocks.SqueezeExcite
class SqueezeExcite(nn.Layer):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=nn.functional.sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv_reduce = nn.Conv2D(in_chs, reduced_chs, 1)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2D(reduced_chs, in_chs, 1)
    
    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

# 原timm.models.effcientnet_blocks.make_divisible
def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
