'''
本文件为原tools/main.py的全复制
本文件为主交互文件
'''

import os
import argparse
import datetime

import _init_paths

from lib.config import cfg

parser = argparse.ArgumentParser(description='Cream of the Crop')
parser.add_argument('mode', type=str, default='train', help='Mode in ["train", "retrain", "test"]')
parser.add_argument('cfg', type=str, default='../experiments/configs/baseline.yaml', help='configuration of cream')
args = parser.parse_args()
cfg.merge_from_file(args.cfg)


def main():
    date = datetime.date.today().strftime('%m%d')
    save_path = os.path.join(cfg.SAVE_PATH, "{}-{}".format(date, cfg.MODEL))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    os.system("copy {} {}".format(args.cfg, os.path.join(save_path, 'config.yaml')))

    if args.mode == 'train':
        os.system("python "
                  # " -m paddle.distributed.launch "
                  "tools/train.py "
                  "--cfg {}".format(args.cfg))
    elif args.mode == 'retrain':
        os.system("python "
                  # "-m paddle.distributed.launch "
                  "tools/retrain.py "
                  "--cfg {}".format(args.cfg))
    elif args.mode == 'test':
        os.system("python "
                  # "-m paddle.distributed.launch "
                  "tools/test.py "
                  "--cfg {}".format(args.cfg))
    else:
        raise ValueError('Mode not supported yet!')


if __name__ == '__main__':
    main()
