<font size = "3">

# 需要转换的代码文件

1. - [x] ~~[Cream\lib\models\blocks\inverted_residual_block.py](Cream\lib\models\blocks\inverted_residual_block.py)~~ - 完成！（97行）检验通过！
2. - [x] ~~[Cream\lib\models\blocks\residual_block.py](Cream\lib\models\blocks\residual_block.py)~~ - 完成！（110行）检验通过！
3. - [ ] ~~[Cream\lib\models\builders\build_childnet.py](Cream\lib\models\builders\build_childnet.py)~~ - 完成！（191行）检验通过！
4. - [ ] ~~[Cream\lib\models\builders\build_supernet.py](Cream\lib\models\builders\build_supernet.py)~~ - 完成！（222行）
5. - [ ] [Cream\lib\models\structures\childnet.py](Cream\lib\models\structures\childnet.py)
6. - [ ] [Cream\lib\models\MetaMatchingNetwork.py](Cream\lib\models\MetaMatchingNetwork.py)
7. - [ ] [Cream\lib\models\PrioritizedBoard.py](Cream\lib\models\PrioritizedBoard.py)
8. - [ ] [Cream\lib\utils\builder_util.py](Cream\lib\utils\builder_util.py)
9. - [ ] [Cream\lib\utils\flops_table.py](Cream\lib\utils\flops_table.py)
10. - [ ] [Cream\lib\utils\util.py](Cream\lib\utils\util.py)
11. - [ ] [Cream\lib\core\train.py](Cream\lib\core\train.py)
12. - [ ] [Cream\lib\core\retrain.py](Cream\lib\core\retrain.py)
13. - [ ] [Cream\lib\core\test.py](Cream\lib\core\test.py)
14. - [ ] [Cream\tools\retrain.py](Cream\tools\retrain.py)

# 需要迁移的库函数/类

- [x] `timm`
  - [x] `models`
    - [x] `efficientnet_blocks`
      - [x] `make_divisible`
      - [x] `round_channels`
    - [x] `layers.get_condconv_initializer`
  - [x] `utils`
    - [x] `accuracy`
    - [x] `AverageMeter`
    - [x] `update_summary`
- [x] `ptflops.get_model_complexity_info`


# 需要重写的库函数/类 -> 完成！

- [x] ~~`timm`~~
  - [x] ~~`data`~~ - 完成！（2+89+196+46+71+275+63=742行）检验通过！
    - [x] ~~`Dataset`~~
    - [x] ~~`create_loader`~~
  - [x] ~~`loss.LabelSmoothingCrossEntropy`~~ - 完成！（1+28=29行）检验通过！
  - [x] ~~`models`~~
    - [x] ~~`efficientnet_blocks`~~ - 完成！（5+98+40+60+246+20+55=524行）检验通过！
      - [x] ~~`ConvBnAct`~~
      - [x] ~~`DepthwiseSeparableConv`~~
      - [x] ~~`drop_path`~~
      - [x] ~~`InvertedResidual`~~
      - [x] ~~`SqueezeExcite`~~
    - [x] ~~`layers`~~
      - [x] ~~`activations`~~ - 完成！（19行）检验通过！
        - [x] ~~`hard_sigmoid`~~
        - [x] ~~`Swish`~~
      - [x] ~~`create_conv2d`~~ - 完成！检验通过！
      - [x] ~~`SelectAdaptivePool2d`~~ - 完成！（88行）检验通过！
    - [x] ~~`resume_checkpoint`~~ - 完成！（33行）检验通过！
  - [x] ~~`optim.create_optimizer`~~ - 完成！（87行）不作检验
  - [x] ~~`scheduler.create_scheduler`~~ - 完成！（1+23=24行）检验通过！
  - [x] ~~`utils`~~ - 完成！（190行）检验通过！
    - [x] ~~`CheckpointSaver`~~
    - [x] ~~`ModelEma`~~
    - [x] ~~`reduce_tensor`~~
- [x] ~~`thop.profile`~~ - 完成！（1+98+110=209行）检验通过！

# 可以忽略差异直接进行迁移的代码文件

1. - [ ] [Cream\lib\utils\op_by_layer_dict.py](Cream\lib\utils\op_by_layer_dict.py)
2. - [ ] [Cream\lib\utils\search_structure_supernet.py](Cream\lib\utils\search_structure_supernet.py)
3. - [ ] [Cream\lib\config.py](Cream\lib\config.py)
4. - [ ] [Cream\tools\_init_paths.py](Cream\tools\_init_paths.py)
5. - [ ] [Cream\tools\generate_subImageNet.py](Cream\tools\generate_subImageNet.py)
6. - [ ] [Cream\tools\main.py](Cream\tools\main.py)

（4、5可能无需迁移）