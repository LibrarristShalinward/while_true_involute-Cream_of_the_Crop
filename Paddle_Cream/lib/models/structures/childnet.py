'''
本文件为原lib/models/structures/childnet.py的转写
'''

import paddle.nn as nn

from ...utils.builder_util import decode_arch_def, efficientnet_init_weights
from ...utils.pimm.models import SelectAdaptivePool2D
from ...utils.pimm.models.activations import Swish, hard_sigmoid
from ...utils.pimm.models.efficientnet_blocks import (create_conv2d,
                                                      resolve_bn_args,
                                                      round_channels)
from ..builders.build_childnet import ChildNetBuilder


# ChildNet Structures
class ChildNet(nn.Layer):

    def __init__(self,
            block_args,
            num_classes = 1000,
            in_chans = 3,
            stem_size = 16,
            num_features = 1280,
            head_bias = True,
            channel_multiplier = 1.0,
            pad_type = '',
            act_layer = nn.ReLU,
            drop_rate = 0.,
            drop_path_rate = 0.,
            se_kwargs = None,
            norm_layer = nn.BatchNorm2D,
            norm_kwargs = None,
            global_pool = 'avg',
            logger = None,
            verbose = False):
        super(ChildNet, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self._in_chs = in_chans
        self.logger = logger

        # Stem
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = create_conv2d(
            self._in_chs, stem_size, 
            3, stride = 2, padding = pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer()
        self._in_chs = stem_size

        # Middle stages (IR/ER/DS Blocks)
        builder = ChildNetBuilder(
            channel_multiplier, 
            8, 
            None, 
            32, 
            pad_type, 
            act_layer, 
            se_kwargs, 
            norm_layer, 
            norm_kwargs, 
            drop_path_rate, 
            verbose = verbose)
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        # self.blocks = builder(self._in_chs, block_args)
        self._in_chs = builder.in_chs

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2D(pool_type = global_pool)
        self.conv_head = create_conv2d(
            self._in_chs,
            self.num_features,
            1,
            padding = pad_type,
            bias_attr = None if head_bias else False)
        self.act2 = act_layer()

        # Classifier
        self.classifier = nn.Linear(
            self.num_features *
            self.global_pool.feat_mult(),
            self.num_classes)

        efficientnet_init_weights(self)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool = 'avg'):
        self.global_pool = SelectAdaptivePool2D(pool_type = global_pool)
        self.num_classes = num_classes
        self.classifier = nn.Linear(
            self.num_features * self.global_pool.feat_mult(),
            num_classes) if self.num_classes else None

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(1)
        if self.drop_rate > 0.:
            x = nn.functional.dropout(x, 
                p = self.drop_rate, 
                training = self.training)
        x = self.classifier(x)
        return x


def gen_childnet(arch_list, arch_def, **kwargs):
    choices = {'kernel_size': [3, 5, 7], 'exp_ratio': [4, 6]}
    choices_list = [[x, y] for x in choices['kernel_size']
                    for y in choices['exp_ratio']]

    num_features = 1280

    act_layer = Swish

    new_arch = []
    # change to child arch_def
    for _, (layer_choice, layer_arch) in enumerate(zip(arch_list, arch_def)):
        if len(layer_arch) == 1:
            new_arch.append(layer_arch)
            continue
        else:
            new_layer = []
            for _, (block_choice, block_arch) in enumerate(
                    zip(layer_choice, layer_arch)):
                kernel_size, exp_ratio = choices_list[block_choice]
                elements = block_arch.split('_')
                block_arch = block_arch.replace(
                    elements[2], 'k{}'.format(str(kernel_size)))
                block_arch = block_arch.replace(
                    elements[4], 'e{}'.format(str(exp_ratio)))
                new_layer.append(block_arch)
            new_arch.append(new_layer)

    model_kwargs = dict(
        block_args = decode_arch_def(new_arch),
        num_features = num_features,
        stem_size = 16,
        norm_kwargs = resolve_bn_args(kwargs),
        act_layer = act_layer,
        se_kwargs = dict(
            act_layer = nn.ReLU,
            gate_fn = hard_sigmoid,
            reduce_mid = True,
            divisor = 8),
        **kwargs,)
    model = ChildNet(**model_kwargs)
    return model
