'''
本文件为原lib/utils/util.py的转写
'''

from .pimm import AverageMeter, accuracy
import paddle
import logging
import sys
from paddle.optimizer import Momentum, Adam
from paddle.optimizer.lr import LambdaDecay
import argparse
from ..config import cfg
from .phop import profile, clever_format
from copy import deepcopy
from paddle.nn import LogSoftmax


def get_path_acc(model, path, val_loader, args, val_iters = 50):
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()
    with paddle.no_grad():
        for batch_idx, (input, target) in enumerate(val_loader):
            if batch_idx >= val_iters:
                break
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            output = model(input, path)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(
                    0,
                    reduce_factor,
                    reduce_factor).mean(
                    dim=2)
                target = target[0:target.size(0):reduce_factor]

            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            # torch.cuda.synchronize()

            prec1_m.update(prec1.item(), output.size(0))
            prec5_m.update(prec5.item(), output.size(0))

    return (prec1_m.avg, prec5_m.avg)


def get_logger(file_path):
    """ Make python logger """
    log_format = '%(asctime)s | %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    logger = logging.getLogger('')

    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def add_weight_decay_supernet(model, args, weight_decay = 1e-5, skip_list = ()):
    decay = []
    no_decay = []
    meta_layer_no_decay = []
    meta_layer_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(
                ".bias") or name in skip_list:
            if 'meta_layer' in name:
                meta_layer_no_decay.append(param)
            else:
                no_decay.append(param)
        else:
            if 'meta_layer' in name:
                meta_layer_decay.append(param)
            else:
                decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0., 'lr': args.lr},
        {'params': decay, 'weight_decay': weight_decay, 'lr': args.lr},
        {'params': meta_layer_no_decay, 'weight_decay': 0., 'lr': args.meta_lr},
        {'params': meta_layer_decay, 'weight_decay': 0, 'lr': args.meta_lr},
    ]


def create_optimizer_supernet(args, model, has_apex, filter_bias_and_bn = True):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    if 'adamw' in opt_lower or 'radam' in opt_lower:
        weight_decay /= args.lr
    if weight_decay and filter_bias_and_bn:
        parameters = add_weight_decay_supernet(model, args, weight_decay)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        # assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'
        assert False, 'APEX and CUDA required for fused optimizers, but CUDA is banned in this version'

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov' or opt_lower == 'momentum':
        optimizer = Momentum(
            parameters = parameters,
            momentum = args.momentum,
            weight_decay = weight_decay)
    elif opt_lower == 'adam':
        optimizer = Adam(
            parameters = parameters, 
            weight_decay = weight_decay, 
            epsilons = args.opt_eps)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer


def convert_lowercase(cfg):
    keys = cfg.keys()
    lowercase_keys = [key.lower() for key in keys]
    values = [cfg.get(key) for key in keys]
    for lowercase_key, value in zip(lowercase_keys, values):
        cfg.setdefault(lowercase_key, value)
    return cfg


def parse_config_args(exp_name):
    parser = argparse.ArgumentParser(description=exp_name)
    parser.add_argument('--cfg', type=str,
                        default='../experiments/workspace/retrain/retrain.yaml',
                        help='configuration of cream')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local_rank')
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    converted_cfg = convert_lowercase(cfg)

    return args, converted_cfg


def get_model_flops_params(model, input_size=(1, 3, 224, 224)):
    input = paddle.randn(input_size)
    macs, params = profile(deepcopy(model), inputs=(input,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params


def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = LogSoftmax(axis = 1)
    return paddle.mean(paddle.sum(- soft_target * logsoftmax(pred), 1))


def create_supernet_scheduler(cfg):
    ITERS = cfg.EPOCHS * \
        (1280000 / (cfg.NUM_GPU * cfg.DATASET.BATCH_SIZE))
    lr_scheduler = LambdaDecay(
        learning_rate = cfg.LR, 
        lr_lambda = lambda step: (
        cfg.LR - step / ITERS) if step <= ITERS else 0, 
        last_epoch = -1)
    return lr_scheduler, cfg.EPOCHS
