# 需要转换的代码文件

1. [Cream\lib\models\blocks\inverted_residual_block.py](Cream\lib\models\blocks\inverted_residual_block.py)
2. [Cream\lib\models\blocks\residual_block.py](Cream\lib\models\blocks\residual_block.py)
3. [Cream\lib\models\builders\build_childnet.py](Cream\lib\models\builders\build_childnet.py)
4. [Cream\lib\models\builders\build_supernet.py](Cream\lib\models\builders\build_supernet.py)
5. [Cream\lib\models\structures\childnet.py](Cream\lib\models\structures\childnet.py)
6. [Cream\lib\models\MetaMatchingNetwork.py](Cream\lib\models\MetaMatchingNetwork.py)
7. [Cream\lib\models\PrioritizedBoard.py](Cream\lib\models\PrioritizedBoard.py)
8. [Cream\lib\utils\builder_util.py](Cream\lib\utils\builder_util.py)
9. [Cream\lib\utils\flops_table.py](Cream\lib\utils\flops_table.py)
10. [Cream\lib\utils\util.py](Cream\lib\utils\util.py)
11. [Cream\lib\core\train.py](Cream\lib\core\train.py)
12. [Cream\lib\core\retrain.py](Cream\lib\core\retrain.py)
13. [Cream\lib\core\test.py](Cream\lib\core\test.py)

# 需要迁移的库函数/类

- `timm.models.efficientnet_blocks`
  - `make_divisible`
  - `round_channels`
- `timm.models.layers`
  - `get_condconv_initializer`
- `ptflops.get_model_complexity_info`
- `timm.utils`
  - `AverageMeter`
  - `accuracy`


# 需要重写的库函数/类

- `timm.models.layers`
  - `create_conv2d`
  - `SelectAdaptivePool2d`
  - `activations.hard_sigmoid`
  - `activations.Swish`
- `timm.models.efficientnet_blocks`
  - `SqueezeExcite`
  - `drop_path`
  - `InvertedResidual`
  - `DepthwiseSeparableConv`
  - `ConvBnAct`
- `thop.profile`
- `timm.utils`
  - `reduce_tensor`

# 可以忽略的代码文件

1. [Cream\lib\utils\op_by_layer_dict.py](Cream\lib\utils\op_by_layer_dict.py)
2. [Cream\lib\utils\search_structure_supernet.py](Cream\lib\utils\search_structure_supernet.py)
3. [Cream\lib\config.py](Cream\lib\config.py)