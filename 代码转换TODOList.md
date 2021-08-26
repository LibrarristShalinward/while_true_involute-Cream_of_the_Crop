<font size = "3">

# 需要转换的代码文件

1. - [ ] [Cream\lib\models\blocks\inverted_residual_block.py](Cream\lib\models\blocks\inverted_residual_block.py)
2. - [ ] [Cream\lib\models\blocks\residual_block.py](Cream\lib\models\blocks\residual_block.py)
3. - [ ] [Cream\lib\models\builders\build_childnet.py](Cream\lib\models\builders\build_childnet.py)
4. - [ ] [Cream\lib\models\builders\build_supernet.py](Cream\lib\models\builders\build_supernet.py)
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

- [ ] `timm`
  - [ ] `models`
    - [ ] `efficientnet_blocks`
      - [ ] `make_divisible`
      - [ ] `round_channels`
    - [ ] `layers.get_condconv_initializer`
  - [ ] `utils`
    - [ ] `accuracy`
    - [ ] `AverageMeter`
    - [ ] `update_summary`
- [ ] `ptflops.get_model_complexity_info`


# 需要重写的库函数/类

- [ ] `timm`
  - [x] ~~`data`~~ - 完成！（2+89+196+46+71+275+63=742行）检验通过！
    - [x] ~~`Dataset`~~
    - [x] ~~`create_loader`~~
  - [x] ~~`loss.LabelSmoothingCrossEntropy`~~ - 完成！（1+28=29行）检验通过！
  - [ ] `models`
    - [ ] ~~`efficientnet_blocks`~~ - 完成！（1+98+40+60+246+20+55=520行）
      - [x] ~~`ConvBnAct`~~
      - [x] ~~`DepthwiseSeparableConv`~~
      - [ ] ~~`drop_path`~~
      - [ ] ~~`InvertedResidual`~~
      - [ ] ~~`SqueezeExcite`~~
    - [ ] `layers`
      - [ ] `activations`
        - [ ] `hard_sigmoid`
        - [ ] `Swish`
      - [ ] ~~`create_conv2d`~~
      - [ ] `SelectAdaptivePool2d`
    - [ ] `resume_checkpoint`
  - [ ] `optim.create_optimizer`
  - [ ] `scheduler.create_scheduler`
  - [ ] `utils`
    - [ ] `CheckpointSaver`
    - [ ] `ModelEma`
      - [ ] `reduce_tensor`
- [ ] `thop.profile`

# 可以忽略差异直接进行迁移的代码文件

1. - [ ] [Cream\lib\utils\op_by_layer_dict.py](Cream\lib\utils\op_by_layer_dict.py)
2. - [ ] [Cream\lib\utils\search_structure_supernet.py](Cream\lib\utils\search_structure_supernet.py)
3. - [ ] [Cream\lib\config.py](Cream\lib\config.py)
4. - [ ] [Cream\tools\_init_paths.py](Cream\tools\_init_paths.py)
5. - [ ] [Cream\tools\generate_subImageNet.py](Cream\tools\generate_subImageNet.py)
6. - [ ] Cream\tools\main.py](Cream\tools\main.py)

（4、5可能无需迁移）