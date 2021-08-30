'''
本文件为timm.models.layers.helpers的全复制修改
由于引用过于复杂，故不对未被调用的函数进行筛除
'''

from itertools import repeat


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        # if isinstance(x, container_abcs.Iterable):
        #     return x
        return tuple(repeat(x, n))
    return parse


tup_single = _ntuple(1)
tup_pair = _ntuple(2)
tup_triple = _ntuple(3)
tup_quadruple = _ntuple(4)
