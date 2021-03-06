'''
本文件为原lib/models/structures/supernet.py的转写
'''

import paddle.nn as nn

from ...utils.builder_util import decode_arch_def, efficientnet_init_weights
from ...utils.op_by_layer_dict import flops_op_dict
from ...utils.pimm.models import SelectAdaptivePool2D
from ...utils.pimm.models.activations import Swish, hard_sigmoid
from ...utils.pimm.models.efficientnet_blocks import (create_conv2d,
                                                      resolve_bn_args,
                                                      round_channels)
from ...utils.search_structure_supernet import search_for_layer
from ..builders.build_supernet import SuperNetBuilder


# Supernet Structures
class SuperNet(nn.Layer):

    def __init__(
            self,
            block_args,
            choices,
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
            slice = 4,
            se_kwargs = None,
            norm_layer = nn.BatchNorm2D,
            logger = None,
            norm_kwargs = None,
            global_pool = 'avg',
            resunit = False,
            dil_conv = False,
            verbose = False):
        super(SuperNet, self).__init__()

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
        builder = SuperNetBuilder(
            choices,
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
            verbose = verbose,
            resunit = resunit,
            dil_conv = dil_conv,
            logger = self.logger)
        self.blocks = builder(self._in_chs, block_args)
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

        self.meta_layer = nn.Linear(self.num_classes * slice, 1)
        efficientnet_init_weights(self)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool = 'avg'):
        self.global_pool = SelectAdaptivePool2D(pool_type = global_pool)
        self.num_classes = num_classes
        self.classifier = nn.Linear(
            self.num_features * self.global_pool.feat_mult(),
            num_classes) if self.num_classes else None

    def forward_features(self, x, architecture):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        for layer, layer_arch in zip(self.blocks, architecture):
            for blocks, arch in zip(layer, layer_arch):
                if arch == -1:
                    continue
                x = blocks[arch](x)

        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x

    def forward(self, x, architecture):
        x = self.forward_features(x, architecture)
        x = x.flatten(1)
        if self.drop_rate > 0.:
            x = nn.functional.dropout(x, p = self.drop_rate, training = self.training)
        return self.classifier(x)

    def forward_meta(self, features):
        return self.meta_layer(features.reshape((1, -1)))

    def rand_parameters(self, architecture, meta = False):
        for name, param in self.named_parameters(include_sublayers = True):
            if 'meta' in name and meta:
                yield param
            elif 'blocks' not in name and 'meta' not in name and (not meta):
                yield param

        if not meta:
            for layer, layer_arch in zip(self.blocks, architecture):
                for blocks, arch in zip(layer, layer_arch):
                    if arch == -1:
                        continue
                    for name, param in blocks[arch].named_parameters(
                            include_sublayers = True):
                        yield param


class Classifier(nn.Layer):
    def __init__(self, num_classes=1000):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        return self.classifier(x)


def gen_supernet(flops_minimum=0, flops_maximum=600, **kwargs):
    choices = {'kernel_size': [3, 5, 7], 'exp_ratio': [4, 6]}

    num_features = 1280

    act_layer = Swish
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25', 'ir_r1_k3_s1_e4_c24_se0.25',
         'ir_r1_k3_s1_e4_c24_se0.25'],
        # stage 2, 56x56 in
        ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r1_k5_s1_e4_c40_se0.25', 'ir_r1_k5_s2_e4_c40_se0.25',
         'ir_r1_k5_s2_e4_c40_se0.25'],
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25', 'ir_r1_k3_s1_e4_c80_se0.25',
         'ir_r2_k3_s1_e4_c80_se0.25'],
        # stage 4, 14x14in
        ['ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25', 'ir_r1_k3_s1_e6_c96_se0.25',
         'ir_r1_k3_s1_e6_c96_se0.25'],
        # stage 5, 14x14in
        ['ir_r1_k5_s2_e6_c192_se0.25', 'ir_r1_k5_s1_e6_c192_se0.25', 'ir_r1_k5_s2_e6_c192_se0.25',
         'ir_r1_k5_s2_e6_c192_se0.25'],
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c320_se0.25'],]

    sta_num, arch_def, resolution = search_for_layer(flops_op_dict, arch_def, flops_minimum, flops_maximum)

    if sta_num is None or arch_def is None or resolution is None:
        raise ValueError('Invalid FLOPs Settings')

    model_kwargs = dict(
        block_args = decode_arch_def(arch_def),
        choices = choices,
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
    model = SuperNet(**model_kwargs)
    return model, sta_num, resolution
