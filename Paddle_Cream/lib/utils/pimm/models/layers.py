from paddle import nn
import paddle
from .conv2d_same import create_conv2d_pad


# 原timm.models.layers.create_conv2d.create_conv2d
def create_conv2d(in_chs, out_chs, kernel_size, **kwargs):
    assert 'groups' not in kwargs
    if isinstance(kernel_size, list):
        assert 'num_experts' not in kwargs  # MixNet + CondConv combo not supported currently
        # We're going to use only lists for defining the MixedConv2d kernel groups,
        # ints, tuples, other iterables will continue to pass to normal conv and specify h, w.
        m = MixedConv2d(in_chs, out_chs, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = out_chs if depthwise else 1
        if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
            m = CondConv2d(in_chs, out_chs, kernel_size, groups=groups, **kwargs)
        else:
            m = create_conv2d_pad(in_chs, out_chs, kernel_size, groups=groups, **kwargs)
    return m

# 原timm.models.layers.mixed_conv2d.MixedConv2d
class MixedConv2d(nn.LayerDict):
    def __init__(
        self, in_channels, out_channels, 
        kernel_size = 3, stride = 1, padding = "", dilation = 1, 
        depthwise = False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)

        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = out_ch if depthwise else 1
            # use add_module to keep key space clean
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits
    
    def forward(self, x):
        x_split = paddle.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        x = paddle.concat(x_out, 1)
        return x

# 原timm.models.layers.mixed_conv2d._split_channels
def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split

