'''
本文件为原tools/train.py的转写
本文件为训练交互文件
'''

import datetime
import os
import sys

import numpy as np
import paddle

import _init_paths
from lib.config import (DEFAULT_CROP_PCT, IMAGENET_DEFAULT_MEAN,
                        IMAGENET_DEFAULT_STD)
from lib.core.train import train_epoch, validate
from lib.models.MetaMatchingNetwork import MetaMatchingNetwork
from lib.models.PrioritizedBoard import PrioritizedBoard
from lib.models.structures.supernet import gen_supernet
from lib.utils.flops_table import FlopsEst
from lib.utils.pimm.data import Dataset, create_loader
from lib.utils.pimm.loss import LabelSmoothingCrossEntropy
from lib.utils.pimm.models import resume_checkpoint
from lib.utils.pimm.utils import CheckpointSaver, update_summary
from lib.utils.util import (create_optimizer_supernet,
                            create_supernet_scheduler, get_logger,
                            parse_config_args)
from paddle.nn import CrossEntropyLoss, SyncBatchNorm

USE_APEX = False


def main():
    args, cfg = parse_config_args('super net training')

    # resolve logging
    cfg.SUPERNET.PICK_METHOD = "top1"
    output_dir = os.path.join(
        cfg.SAVE_PATH, "{}-{}".format(
            datetime.date.today().strftime('%m%d'), 
            cfg.MODEL))

    if args.local_rank == 0:
        logger = get_logger(os.path.join(output_dir, "train.log"))
    else:
        logger = None

    # initialize distributed parameters
    if args.local_rank == 0:
        logger.info(
            'Training on Process %d with %d GPUs.',
                args.local_rank, cfg.NUM_GPU)

    # fix random seeds
    paddle.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    # generate supernet
    model, sta_num, resolution = gen_supernet(
        flops_minimum = cfg.SUPERNET.FLOPS_MINIMUM,
        flops_maximum = cfg.SUPERNET.FLOPS_MAXIMUM,
        num_classes = cfg.DATASET.NUM_CLASSES,
        drop_rate = cfg.NET.DROPOUT_RATE,
        global_pool = cfg.NET.GP,
        resunit = cfg.SUPERNET.RESUNIT,
        dil_conv = cfg.SUPERNET.DIL_CONV,
        slice = cfg.SUPERNET.SLICE,
        verbose = cfg.VERBOSE,
        logger = logger)

    # initialize meta matching networks
    MetaMN = MetaMatchingNetwork(cfg)

    # number of choice blocks in supernet
    choice_num = len(model.blocks[1][0])
    if args.local_rank == 0:
        logger.info('Supernet created, param count: %d', (
            sum([m.numel() for m in model.parameters()])))
        logger.info('resolution: %d', (resolution))
        logger.info('choice number: %d', (choice_num))

    #initialize prioritized board
    prioritized_board = PrioritizedBoard(cfg, CHOICE_NUM = choice_num, sta_num = sta_num)

    # initialize flops look-up table
    model_est = FlopsEst(model)

    # optionally resume from a checkpoint
    optimizer_state = None
    resume_epoch = None
    if cfg.AUTO_RESUME:
        optimizer_state, resume_epoch = resume_checkpoint(
            model, cfg.RESUME_PATH)

    # create learning rate scheduler
    lr_scheduler, num_epochs = create_supernet_scheduler(cfg)

    start_epoch = resume_epoch if resume_epoch is not None else 0
    if start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        logger.info('Scheduled epochs: %d', num_epochs)

    # create optimizer and resume from checkpoint
    optimizer = create_optimizer_supernet(cfg, model, lr_scheduler)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state['optimizer'])
    # model = model.cuda()

    # convert model to distributed mode
    if cfg.BATCHNORM.SYNC_BN:
        try:
            if USE_APEX:
                assert False, "APEX is not available!!!"
            else:
                model = SyncBatchNorm.convert_sync_batchnorm(model)
            if args.local_rank == 0:
                logger.info('Converted model to use Synchronized BatchNorm.')
        except Exception as exception:
            logger.info(
                'Failed to enable Synchronized BatchNorm. '
                'Install Apex or Torch >= 1.1 with Exception %s', exception)
    if USE_APEX:
        assert False, "APEX is not available!!!"
    else:
        if args.local_rank == 0:
            logger.info("Using paddle DistributedDataParallel. ")

    # imagenet train dataset
    train_dir = os.path.join(cfg.DATA_DIR, 'train')
    if not os.path.exists(train_dir):
        logger.info('Training folder does not exist at: %s', train_dir)
        sys.exit()

    dataset_train = Dataset(train_dir)
    loader_train = create_loader(
        dataset_train,
        input_size = (3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE),
        batch_size = cfg.DATASET.BATCH_SIZE,
        is_training = True,
        use_prefetcher = True,
        re_prob = cfg.AUGMENTATION.RE_PROB,
        re_mode = cfg.AUGMENTATION.RE_MODE,
        color_jitter = cfg.AUGMENTATION.COLOR_JITTER,
        interpolation = 'random',
        num_workers = cfg.WORKERS,
        distributed = True,
        collate_fn = None,
        crop_pct = DEFAULT_CROP_PCT,
        mean = IMAGENET_DEFAULT_MEAN,
        std = IMAGENET_DEFAULT_STD)

    # imagenet validation dataset
    eval_dir = os.path.join(cfg.DATA_DIR, 'val')
    if not os.path.isdir(eval_dir):
        logger.info('Validation folder does not exist at: %s', eval_dir)
        sys.exit()
    dataset_eval = Dataset(eval_dir)
    loader_eval = create_loader(
        dataset_eval,
        input_size = (3, cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE),
        batch_size = 2 * cfg.DATASET.BATCH_SIZE,
        is_training = False,
        use_prefetcher = True,
        num_workers = cfg.WORKERS,
        distributed = True,
        crop_pct = DEFAULT_CROP_PCT,
        mean = IMAGENET_DEFAULT_MEAN,
        std = IMAGENET_DEFAULT_STD,
        interpolation = cfg.DATASET.INTERPOLATION
    )

    # whether to use label smoothing
    if cfg.AUGMENTATION.SMOOTHING > 0.:
        train_loss_fn = LabelSmoothingCrossEntropy(
            smoothing = cfg.AUGMENTATION.SMOOTHING)
        validate_loss_fn = CrossEntropyLoss()
    else:
        train_loss_fn = CrossEntropyLoss()
        validate_loss_fn = train_loss_fn

    # initialize training parameters
    eval_metric = cfg.EVAL_METRICS
    best_metric, best_epoch, saver, best_children_pool = None, None, None, []
    if args.local_rank == 0:
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            checkpoint_dir = output_dir,
            decreasing = decreasing)

    # training scheme
    try:
        for epoch in range(start_epoch, num_epochs):
            loader_train.sampler.set_epoch(epoch)

            # train one epoch
            train_metrics = train_epoch(epoch, model, loader_train, optimizer,
                                        train_loss_fn, prioritized_board, MetaMN, cfg,
                                        lr_scheduler = lr_scheduler, saver = saver,
                                        output_dir = output_dir, logger = logger,
                                        est = model_est, local_rank = args.local_rank)

            # evaluate one epoch
            eval_metrics = validate(model, loader_eval, validate_loss_fn,
                                    prioritized_board, cfg,
                                    local_rank = args.local_rank, logger = logger)

            update_summary(epoch, train_metrics, eval_metrics, os.path.join(
                output_dir, 'summary.csv'), write_header = best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(
                    model, optimizer, cfg,
                    epoch = epoch, metric = save_metric)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
