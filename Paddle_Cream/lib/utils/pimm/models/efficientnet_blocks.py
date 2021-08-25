from paddle import nn

# 原timm.model.effcientnet_blocks.ConvBnAct
class ConvBnAct(nn.Layer):
    def __init__(
        self, in_chs, out_chs, 
        kernel_size, stride = 1, dilation = 1, pad_type= "", 
        act_layer = nn.ReLU, norm_layer = nn.BatchNorm2D, 
        norm_kwargs = None):
        super(ConvBnAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = 

