'''
本文件为原tools/retrain.py的转写
'''

import os
import datetime
import numpy as np
import paddle
import _init_paths
from paddle.nn import CrossEntropyLoss

# import pimm packages
from lib.utils.pimm.optim import create_optimizer
from lib.utils.pimm.models import resume_checkpoint
from lib.utils.pimm.scheduler import create_scheduler
from lib.utils.pimm.data import Dataset, create_loader
from lib.utils.pimm.utils import ModelEma, update_summary, CheckpointSaver
from lib.utils.pimm.loss import LabelSmoothingCrossEntropy

# import models and training functions
from lib.core.test import validate
from lib.core.retrain import train_epoch
from lib.models.structures.childnet import gen_childnet
from lib.utils.util import parse_config_args, get_logger, get_model_flops_params
from lib.config import DEFAULT_CROP_PCT, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def main():
    args, cfg = parse_config_args('child net training')

    # resolve logging
    output_dir = os.path.join(cfg.SAVE_PATH,
                              "{}-{}".format(datetime.date.today().strftime('%m%d'),
                                             cfg.MODEL))

    if args.local_rank == 0:
        logger = get_logger(os.path.join(output_dir, 'retrain.log'))
    else:
        logger = None
    writer = None

    # retrain model selection
    if cfg.NET.SELECTION == 481:
        arch_list = [
            [0], [
                3, 4, 3, 1], [
                3, 2, 3, 0], [
                3, 3, 3, 1, 1], [
                    3, 3, 3, 3], [
                        3, 3, 3, 3], [0]]
        cfg.DATASET.IMAGE_SIZE = 224
    elif cfg.NET.SELECTION == 43:
        arch_list = [[0], [3], [3, 1], [3, 1], [3, 3, 3], [3, 3], [0]]
        cfg.DATASET.IMAGE_SIZE = 96
    elif cfg.NET.SELECTION == 14:
        arch_list = [[0], [3], [3, 3], [3, 3], [3], [3], [0]]
        cfg.DATASET.IMAGE_SIZE = 64
    elif cfg.NET.SELECTION == 114:
        arch_list = [[0], [3], [3, 3], [3, 3], [3, 3, 3], [3, 3], [0]]
        cfg.DATASET.IMAGE_SIZE = 160
    elif cfg.NET.SELECTION == 287:
        arch_list = [[0], [3], [3, 3], [3, 1, 3], [3, 3, 3, 3], [3, 3, 3], [0]]
        cfg.DATASET.IMAGE_SIZE = 224
    elif cfg.NET.SELECTION == 604:
        arch_list = [
            [0], [
                3, 3, 2, 3, 3], [
                3, 2, 3, 2, 3], [
                3, 2, 3, 2, 3], [
                    3, 3, 2, 2, 3, 3], [
                        3, 3, 2, 3, 3, 3], [0]]
        cfg.DATASET.IMAGE_SIZE = 224
    else:
        raise ValueError("Model Retrain Selection is not Supported!")

    # define childnet architecture from arch_list
    stem = ['ds_r1_k3_s1_e1_c16_se0.25', 'cn_r1_k1_s1_c320_se0.25']
    choice_block_pool = ['ir_r1_k3_s2_e4_c24_se0.25',
                         'ir_r1_k5_s2_e4_c40_se0.25',
                         'ir_r1_k3_s2_e6_c80_se0.25',
                         'ir_r1_k3_s1_e6_c96_se0.25',
                         'ir_r1_k5_s2_e6_c192_se0.25']
    arch_def = [[stem[0]]] + [[choice_block_pool[idx]
                               for repeat_times in range(len(arch_list[idx + 1]))]
                              for idx in range(len(choice_block_pool))] + [[stem[1]]]

    # generate childnet
    model = gen_childnet(
        arch_list,
        arch_def,
        num_classes = cfg.DATASET.NUM_CLASSES,
        drop_rate = cfg.NET.DROPOUT_RATE,
        global_pool = cfg.NET.GP)

    # initialize training parameters
    eval_metric = cfg.EVAL_METRICS
    best_metric, best_epoch, saver = None, None, None
    if args.local_rank == 0:
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            checkpoint_dir = output_dir,
            decreasing = decreasing)

    # initialize distributed parameters
    distributed = cfg.NUM_GPU > 1
    if args.local_rank == 0:
        logger.info(
            'Training on Process {} with {} GPUs.'.format(
                args.local_rank, cfg.NUM_GPU))

    # fix random seeds
    paddle.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    # get parameters and FLOPs of model
    if args.local_rank == 0:
        macs, params = get_model_flops_params(model, input_size = (
            1, 3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE))
        logger.info(
            '[Model-{}] Flops: {} Params: {}'.format(cfg.NET.SELECTION, macs, params))

    # optionally resume from a checkpoint
    resume_state, resume_epoch = {}, None
    if cfg.AUTO_RESUME:
        resume_state, resume_epoch = resume_checkpoint(model, cfg.RESUME_PATH)

    # create learning rate scheduler
    lr_scheduler, num_epochs = create_scheduler(cfg)
    start_epoch = resume_epoch if resume_epoch is not None else 0
    if start_epoch > 0:
        lr_scheduler.step(start_epoch)
    if args.local_rank == 0:
        logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create optimizer
    optimizer = create_optimizer(cfg, model, lr_scheduler, False)
    if cfg.AUTO_RESUME:
        optimizer.load_state_dict(resume_state['optimizer'])
        del resume_state

    model_ema = None
    if cfg.NET.EMA.USE:
        model_ema = ModelEma(
            model,
            decay = cfg.NET.EMA.DECAY,
            device = 'cpu' if cfg.NET.EMA.FORCE_CPU else '',
            resume = cfg.RESUME_PATH if cfg.AUTO_RESUME else None)

    if distributed:
        assert False, "Distributed not available! GPU num: " + str(cfg.NUM_GPU)

    # imagenet train dataset
    train_dir = os.path.join(cfg.DATA_DIR, 'train')
    if not os.path.exists(train_dir) and args.local_rank == 0:
        logger.error('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    dataset_train = Dataset(train_dir)
    loader_train = create_loader(
        dataset_train,
        input_size = (3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE),
        batch_size = cfg.DATASET.BATCH_SIZE,
        is_training = True,
        color_jitter = cfg.AUGMENTATION.COLOR_JITTER,
        auto_augment = cfg.AUGMENTATION.AA,
        num_aug_splits = 0,
        crop_pct = DEFAULT_CROP_PCT,
        mean = IMAGENET_DEFAULT_MEAN,
        std = IMAGENET_DEFAULT_STD,
        num_workers = cfg.WORKERS,
        distributed = True,
        collate_fn = None,
        interpolation = 'random',
        re_mode = cfg.AUGMENTATION.RE_MODE,
        re_prob = cfg.AUGMENTATION.RE_PROB)

    # imagenet validation dataset
    eval_dir = os.path.join(cfg.DATA_DIR, 'val')
    if not os.path.exists(eval_dir) and args.local_rank == 0:
        logger.error(
            'Validation folder does not exist at: {}'.format(eval_dir))
        exit(1)
    dataset_eval = Dataset(eval_dir)
    loader_eval = create_loader(
        dataset_eval,
        input_size = (3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE),
        batch_size = cfg.DATASET.VAL_BATCH_MUL * cfg.DATASET.BATCH_SIZE,
        is_training = False,
        interpolation = 'bicubic',
        crop_pct = DEFAULT_CROP_PCT,
        mean = IMAGENET_DEFAULT_MEAN,
        std = IMAGENET_DEFAULT_STD,
        num_workers = cfg.WORKERS,
        distributed = True)

    # whether to use label smoothing
    if cfg.AUGMENTATION.SMOOTHING > 0.:
        train_loss_fn = LabelSmoothingCrossEntropy(
            smoothing = cfg.AUGMENTATION.SMOOTHING)
        validate_loss_fn = CrossEntropyLoss()
    else:
        train_loss_fn = CrossEntropyLoss()
        validate_loss_fn = train_loss_fn

    

    try:
        best_record, best_ep = 0, 0
        for epoch in range(start_epoch, num_epochs):
            if distributed:
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                cfg,
                lr_scheduler = lr_scheduler,
                saver = saver,
                output_dir = output_dir,
                model_ema = model_ema,
                logger = logger,
                writer = writer,
                local_rank = args.local_rank)

            eval_metrics = validate(
                epoch,
                model,
                loader_eval,
                validate_loss_fn,
                cfg,
                logger = logger,
                writer = writer,
                local_rank = args.local_rank)

            if model_ema is not None and not cfg.NET.EMA.FORCE_CPU:
                ema_eval_metrics = validate(
                    epoch,
                    model_ema.ema,
                    loader_eval,
                    validate_loss_fn,
                    cfg,
                    log_suffix = '_EMA',
                    logger = logger,
                    writer = writer,
                    local_rank = args.local_rank)
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)

            update_summary(epoch, train_metrics, eval_metrics, os.path.join(
                output_dir, 'summary.csv'), write_header = best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    model, optimizer, cfg,
                    epoch = epoch, model_ema = model_ema, metric = save_metric)

            if best_record < eval_metrics[eval_metric]:
                best_record = eval_metrics[eval_metric]
                best_ep = epoch

            if args.local_rank == 0:
                logger.info(
                    '*** Best metric: {0} (epoch {1})'.format(best_record, best_ep))

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        logger.info(
            '*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


if __name__ == '__main__':
    main()
