from paddle import nn
import paddle.nn.functional as F


# 原timm.models.layers.activations.hard_sigmoid
def hard_sigmoid(x):
    return F.relu6(x + 3.) / 6.

# 原timm.models.layers.activations.Swish
class Swish(nn.Layer):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return swish(x)

# 原timm.models.layers.activations.swish（非inplace版本）
def swish(x):
    return x * F.sigmoid(x)
