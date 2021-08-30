'''
本文件为原lib/utils/builder_util.py的转写
'''



def modify_block_args(block_args, kernel_size, exp_ratio):
    block_type = block_args['block_type']
    if block_type == 'cn':
        block_args['kernel_size'] = kernel_size
    elif block_type == 'er':
        block_args['exp_kernel_size'] = kernel_size
    else:
        block_args['dw_kernel_size'] = kernel_size

    if block_type == 'ir' or block_type == 'er':
        block_args['exp_ratio'] = exp_ratio
    return block_args