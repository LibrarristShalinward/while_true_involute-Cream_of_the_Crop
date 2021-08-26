from Paddle_Cream.lib.utils.pimm.models.create_conv2d import create_conv2d
from paddle import nn
from .cond_conv2d import create_conv2d

# 原timm.model.effcientnet_blocks.ConvBnAct
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
        self.act1 = act_layer(inplace=True)
    
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