'''
本文件为原lib/utils/flops_table.py的转写
'''

import paddle

from .pdflops import get_model_complexity_info


class FlopsEst(object):
    def __init__(self, 
        model, 
        input_shape = (2, 3, 224, 224), 
        device = 'cpu'):
        self.block_num = len(model.blocks)
        self.choice_num = len(model.blocks[0])
        self.flops_dict = {}
        self.params_dict = {}

        self.params_fixed = 0
        self.flops_fixed = 0

        input = paddle.randn(input_shape)

        flops, params = get_model_complexity_info(
            model.conv_stem, 
            (3, 224, 224), 
            as_strings = False, 
            print_per_layer_stat = False)
        self.params_fixed += params / 1e6
        self.flops_fixed += flops / 1e6

        input = model.conv_stem(input)

        for block_id, block in enumerate(model.blocks):
            self.flops_dict[block_id] = {}
            self.params_dict[block_id] = {}
            for module_id, module in enumerate(block):
                self.flops_dict[block_id][module_id] = {}
                self.params_dict[block_id][module_id] = {}
                for choice_id, choice in enumerate(module):
                    flops, params = get_model_complexity_info(
                        choice, 
                        tuple(input.shape[1:]), 
                        as_strings = False, 
                        print_per_layer_stat = False)
                    # Flops(M)
                    self.flops_dict[block_id][module_id][choice_id] = flops / 1e6
                    # Params(M)
                    self.params_dict[block_id][module_id][choice_id] = params / 1e6

                input = choice(input)

        # conv_last
        flops, params = get_model_complexity_info(
            model.global_pool, 
            tuple(input.shape[1:]), 
            as_strings = False, 
            print_per_layer_stat = False)
        self.params_fixed += params / 1e6
        self.flops_fixed += flops / 1e6

        input = model.global_pool(input)

        # globalpool
        flops, params = get_model_complexity_info(
            model.conv_head, 
            tuple(input.shape[1:]), 
            as_strings = False, 
            print_per_layer_stat = False)
        self.params_fixed += params / 1e6
        self.flops_fixed += flops / 1e6

    # return params (M)
    def get_params(self, arch):
        params = 0
        for block_id, block in enumerate(arch):
            for module_id, choice in enumerate(block):
                if choice == -1:
                    continue
                params += self.params_dict[block_id][module_id][choice]
        return params + self.params_fixed

    # return flops (M)
    def get_flops(self, arch):
        flops = 0
        for block_id, block in enumerate(arch):
            for module_id, choice in enumerate(block):
                if choice == -1:
                    continue
                flops += self.flops_dict[block_id][module_id][choice]
        return flops + self.flops_fixed

