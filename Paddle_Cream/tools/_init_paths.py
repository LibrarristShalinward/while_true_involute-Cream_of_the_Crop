'''
本文件为原tools/_init_paths.py的全复制
本文件为环境初始化文件
'''

from __future__ import absolute_import, division, print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

lib_path = osp.join(this_dir, '..')
add_path(lib_path)
