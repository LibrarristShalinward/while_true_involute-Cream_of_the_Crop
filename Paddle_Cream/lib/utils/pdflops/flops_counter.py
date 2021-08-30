'''
本文件为ptflops.flops_counter的全复制修改
由于引用过于复杂，故不对未被调用的函数进行筛除
'''

import sys
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn


def get_model_complexity_info(model, input_res,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None, ost=sys.stdout,
                              verbose=False, ignore_layers=[],
                              custom_layers_hooks={}):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Layer)
    global CUSTOM_LAYERS_MAPPING
    CUSTOM_LAYERS_MAPPING = custom_layers_hooks
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count(ost=ost, verbose=verbose,
                                  ignore_list=ignore_layers)
    if input_constructor:
        input = input_constructor(input_res)
        _ = flops_model(**input)
    else:
        try:
            batch = paddle.randn(
                (1, *input_res), 
                dtype=flops_model.parameters()[0].dtype)
        except StopIteration:
            batch = paddle.randn((1, *input_res))

        _ = flops_model(batch)

    flops_count, params_count = flops_model.compute_average_flops_cost()
    if print_per_layer_stat:
        print_model_with_flops(flops_model, flops_count, params_count, ost=ost)
    flops_model.stop_flops_count()
    CUSTOM_LAYERS_MAPPING = {}

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


def flops_to_string(flops, units='GMac', precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, 2)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, 2)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)


def accumulate_flops(self):
    if is_supported_instance(self):
        return self.__flops__
    else:
        sum = 0
        for m in self.children():
            sum += m.accumulate_flops()
        return sum


def print_model_with_flops(model, total_flops, total_params, units='GMac',
                           precision=3, ost=sys.stdout):
    if total_flops < 1:
        total_flops = 1

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def flops_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops() / model.__batch_counter__
        return ', '.join([params_to_string(accumulated_params_num,
                                           units='M', precision=precision),
                          '{:.3%} Params'.format(accumulated_params_num / total_params),
                          flops_to_string(accumulated_flops_cost,
                                          units=units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_flops_cost / total_flops),
                          self.original_extra_repr()])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(repr(model), file=ost)
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
    return params_num


def add_flops_counting_methods(net_main_layer):
    # adding additional methods to the existing layer object,
    # this is done this way so that each function has access to self object
    net_main_layer.start_flops_count = start_flops_count.__get__(net_main_layer)
    net_main_layer.stop_flops_count = stop_flops_count.__get__(net_main_layer)
    net_main_layer.reset_flops_count = reset_flops_count.__get__(net_main_layer)
    net_main_layer.compute_average_flops_cost = compute_average_flops_cost.__get__(
                                                    net_main_layer)

    net_main_layer.reset_flops_count()

    return net_main_layer


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Returns current mean flops consumption per image.
    """

    for m in self.sublayers(include_self = True):
        m.accumulate_flops = accumulate_flops.__get__(m)

    flops_sum = self.accumulate_flops()

    for m in self.sublayers(include_self = True):
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    params_sum = get_model_parameters_number(self)
    return flops_sum / self.__batch_counter__, params_sum


def start_flops_count(self, **kwargs):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Activates the computation of mean flops consumption per image.
    Call it before you run the network.
    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    def add_flops_counter_hook_function(layer, ost, verbose, ignore_list):
        if type(layer) in ignore_list:
            seen_types.add(type(layer))
            if is_supported_instance(layer):
                layer.__params__ = 0
        elif is_supported_instance(layer):
            if hasattr(layer, '__flops_handle__'):
                return
            if type(layer) in CUSTOM_LAYERS_MAPPING:
                handle = layer.register_forward_post_hook(
                                        CUSTOM_LAYERS_MAPPING[type(layer)])
            else:
                handle = layer.register_forward_post_hook(LAYERS_MAPPING[type(layer)])
            layer.__flops_handle__ = handle
            seen_types.add(type(layer))
        else:
            if verbose and not type(layer) in (nn.Sequential, nn.LayerList) and \
               not type(layer) in seen_types:
                print('Warning: layer ' + type(layer).__name__ +
                      ' is treated as a zero-op.', file=ost)
            seen_types.add(type(layer))

    self.apply(partial(add_flops_counter_hook_function, **kwargs))


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.
    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Resets statistics computed so far.
    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


# ---- Internal functions
def empty_flops_counter_hook(layer, input, output):
    layer.__flops__ += 0


def upsample_flops_counter_hook(layer, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    layer.__flops__ += int(output_elements_count)


def relu_flops_counter_hook(layer, input, output):
    active_elements_count = output.numel()
    layer.__flops__ += int(active_elements_count)


def linear_flops_counter_hook(layer, input, output):
    input = input[0]
    # pytorch checks dimensions, so here we don't care much
    output_last_dim = output.shape[-1]
    bias_flops = output_last_dim if layer.bias is not None else 0
    layer.__flops__ += int(np.prod(input.shape) * output_last_dim + bias_flops)


def pool_flops_counter_hook(layer, input, output):
    input = input[0]
    layer.__flops__ += int(np.prod(input.shape))


def bn_flops_counter_hook(layer, input, output):
    input = input[0]

    batch_flops = np.prod(input.shape)
    if layer.affine:
        batch_flops *= 2
    layer.__flops__ += int(batch_flops)


def conv_flops_counter_hook(conv_layer, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_layer._kernel_size)
    in_channels = conv_layer._in_channels
    out_channels = conv_layer._out_channels
    groups = conv_layer._groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_layer.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_layer.__flops__ += int(overall_flops)


def batch_counter_hook(layer, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a layer,'
              ' assuming batch size is 1.')
    layer.__batch_counter__ += batch_size


def rnn_flops(flops, rnn_layer, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0]*w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0]*w_hh.shape[1]
    if isinstance(rnn_layer, (nn.RNN, nn.SimpleRNNCell)):
        # add both operations
        flops += rnn_layer.hidden_size
    elif isinstance(rnn_layer, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_layer.hidden_size
        # adding operations from both states
        flops += rnn_layer.hidden_size*3
        # last two hadamard product and add
        flops += rnn_layer.hidden_size*3
    elif isinstance(rnn_layer, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_layer.hidden_size*4
        # two hadamard product and add for C state
        flops += rnn_layer.hidden_size + rnn_layer.hidden_size + rnn_layer.hidden_size
        # final hadamard
        flops += rnn_layer.hidden_size + rnn_layer.hidden_size + rnn_layer.hidden_size
    return flops


def rnn_flops_counter_hook(rnn_layer, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    IF sigmoid and tanh are made hard, only a comparison FLOPS should be accurate
    """
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_layer.num_layers

    for i in range(num_layers):
        w_ih = rnn_layer.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_layer.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_layer.input_size
        else:
            input_size = rnn_layer.hidden_size
        flops = rnn_flops(flops, rnn_layer, w_ih, w_hh, input_size)
        if rnn_layer.bias:
            b_ih = rnn_layer.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_layer.__getattr__('bias_hh_l' + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_layer.bidirectional:
        flops *= 2
    rnn_layer.__flops__ += int(flops)


def rnn_cell_flops_counter_hook(rnn_cell_layer, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_layer.__getattr__('weight_ih')
    w_hh = rnn_cell_layer.__getattr__('weight_hh')
    input_size = inp.shape[1]
    flops = rnn_flops(flops, rnn_cell_layer, w_ih, w_hh, input_size)
    if rnn_cell_layer.bias:
        b_ih = rnn_cell_layer.__getattr__('bias_ih')
        b_hh = rnn_cell_layer.__getattr__('bias_hh')
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_layer.__flops__ += int(flops)


def multihead_attention_counter_hook(multihead_attention_layer, input, output):
    flops = 0
    q, k, v = input
    batch_size = q.shape[1]

    num_heads = multihead_attention_layer.num_heads
    embed_dim = multihead_attention_layer.embed_dim
    kdim = multihead_attention_layer.kdim
    vdim = multihead_attention_layer.vdim
    if kdim is None:
        kdim = embed_dim
    if vdim is None:
        vdim = embed_dim

    # initial projections
    flops = q.shape[0] * q.shape[2] * embed_dim + \
        k.shape[0] * k.shape[2] * kdim + \
        v.shape[0] * v.shape[2] * vdim
    if multihead_attention_layer.in_proj_bias is not None:
        flops += (q.shape[0] + k.shape[0] + v.shape[0]) * embed_dim

    # attention heads: scale, matmul, softmax, matmul
    head_dim = embed_dim // num_heads
    head_flops = q.shape[0] * head_dim + \
        head_dim * q.shape[0] * k.shape[0] + \
        q.shape[0] * k.shape[0] + \
        q.shape[0] * k.shape[0] * head_dim

    flops += num_heads * head_flops

    # final projection, bias is always enabled
    flops += q.shape[0] * embed_dim * (embed_dim + 1)

    flops *= batch_size
    multihead_attention_layer.__flops__ += int(flops)


def add_batch_counter_variables_or_reset(layer):

    layer.__batch_counter__ = 0


def add_batch_counter_hook_function(layer):
    if hasattr(layer, '__batch_counter_handle__'):
        return

    handle = layer.register_forward_post_hook(batch_counter_hook)
    layer.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(layer):
    if hasattr(layer, '__batch_counter_handle__'):
        layer.__batch_counter_handle__.remove()
        del layer.__batch_counter_handle__


def add_flops_counter_variable_or_reset(layer):
    if is_supported_instance(layer):
        if hasattr(layer, '__flops__') or hasattr(layer, '__params__'):
            print('Warning: variables __flops__ or __params__ are already '
                  'defined for the layer' + type(layer).__name__ +
                  ' ptflops can affect your code!')
        layer.__flops__ = 0
        layer.__params__ = get_model_parameters_number(layer)


CUSTOM_LAYERS_MAPPING = {}

LAYERS_MAPPING = {
    # convolutions
    nn.Conv1D: conv_flops_counter_hook,
    nn.Conv2D: conv_flops_counter_hook,
    nn.Conv3D: conv_flops_counter_hook,
    # activations
    nn.ReLU: relu_flops_counter_hook,
    nn.PReLU: relu_flops_counter_hook,
    nn.ELU: relu_flops_counter_hook,
    nn.LeakyReLU: relu_flops_counter_hook,
    nn.ReLU6: relu_flops_counter_hook,
    # poolings
    nn.MaxPool1D: pool_flops_counter_hook,
    nn.AvgPool1D: pool_flops_counter_hook,
    nn.AvgPool2D: pool_flops_counter_hook,
    nn.MaxPool2D: pool_flops_counter_hook,
    nn.MaxPool3D: pool_flops_counter_hook,
    nn.AvgPool3D: pool_flops_counter_hook,
    nn.AdaptiveMaxPool1D: pool_flops_counter_hook,
    nn.AdaptiveAvgPool1D: pool_flops_counter_hook,
    nn.AdaptiveMaxPool2D: pool_flops_counter_hook,
    nn.AdaptiveAvgPool2D: pool_flops_counter_hook,
    nn.AdaptiveMaxPool3D: pool_flops_counter_hook,
    nn.AdaptiveAvgPool3D: pool_flops_counter_hook,
    # BNs
    nn.BatchNorm1D: bn_flops_counter_hook,
    nn.BatchNorm2D: bn_flops_counter_hook,
    nn.BatchNorm3D: bn_flops_counter_hook,

    nn.InstanceNorm1D: bn_flops_counter_hook,
    nn.InstanceNorm2D: bn_flops_counter_hook,
    nn.InstanceNorm3D: bn_flops_counter_hook,
    nn.GroupNorm: bn_flops_counter_hook,
    # FC
    nn.Linear: linear_flops_counter_hook,
    # Upscale
    nn.Upsample: upsample_flops_counter_hook,
    # Deconvolution
    nn.Conv1DTranspose: conv_flops_counter_hook,
    nn.Conv2DTranspose: conv_flops_counter_hook,
    nn.Conv3DTranspose: conv_flops_counter_hook,
    # RNN
    nn.RNN: rnn_flops_counter_hook,
    nn.GRU: rnn_flops_counter_hook,
    nn.LSTM: rnn_flops_counter_hook,
    nn.SimpleRNNCell: rnn_cell_flops_counter_hook,
    nn.LSTMCell: rnn_cell_flops_counter_hook,
    nn.GRUCell: rnn_cell_flops_counter_hook,
    nn.MultiHeadAttention: multihead_attention_counter_hook
}


def is_supported_instance(layer):
    if type(layer) in LAYERS_MAPPING or type(layer) in CUSTOM_LAYERS_MAPPING:
        return True
    return False


def remove_flops_counter_hook_function(layer):
    if is_supported_instance(layer):
        if hasattr(layer, '__flops_handle__'):
            layer.__flops_handle__.remove()
            del layer.__flops_handle__
