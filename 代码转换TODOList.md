# 需要转换的代码文件

- [Cream\lib\models\blocks\inverted_residual_block.py](Cream\lib\models\blocks\inverted_residual_block.py)
- [Cream\lib\models\blocks\residual_block.py](Cream\lib\models\blocks\residual_block.py)
- [Cream\lib\models\builders\build_childnet.py](Cream\lib\models\builders\build_childnet.py)
- [Cream\lib\models\builders\build_supernet.py](Cream\lib\models\builders\build_supernet.py)
- [Cream\lib\models\structures\childnet.py](Cream\lib\models\structures\childnet.py)
- [Cream\lib\models\MetaMatchingNetwork.py](Cream\lib\models\MetaMatchingNetwork.py)
- [Cream\lib\models\PrioritizedBoard.py](Cream\lib\models\PrioritizedBoard.py)

# 需要迁移的库函数/类

- `timm.models.efficientnet_blocks`
  - `make_divisible`
  - `round_channels`


# 需要重写的库函数/类

- `timm.models.layers`
  - `create_conv2d`
  - `SelectAdaptivePool2d`
  - `activations.hard_sigmoid`
- `timm.models.efficientnet_blocks`
  - `SqueezeExcite`
  - `drop_path`
  - `InvertedResidual`
  - `DepthwiseSeparableConv`
  - `ConvBnAct`

# 可以忽略的代码文件