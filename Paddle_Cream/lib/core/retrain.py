'''
本文件为原lib/models/core/retrain.py的转写
'''

import os
import time
from collections import OrderedDict

from lib.utils.pimm.utils import accuracy, reduce_tensor
from lib.utils.util import AverageMeter
from PIL import Image


# retrain function
def train_epoch(
        epoch, 
        model, 
        loader, 
        optimizer, 
        loss_fn, 
        cfg,
        lr_scheduler = None, 
        saver = None, 
        output_dir = '', 
        use_amp = False,
        model_ema = None, 
        logger = None, 
        writer = None, 
        local_rank = 0):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    optimizer.clear_grad()
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        output = model(input)

        loss = loss_fn(output, target)

        prec1, prec5 = accuracy(output, target, topk = (1, 5))

        if cfg.NUM_GPU > 1:
            reduced_loss = reduce_tensor(loss, cfg.NUM_GPU)
            prec1 = reduce_tensor(prec1, cfg.NUM_GPU)
            prec5 = reduce_tensor(prec5, cfg.NUM_GPU)
        else:
            reduced_loss = loss

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        losses_m.update(reduced_loss.item(), input.shape[0])
        prec1_m.update(prec1, output.shape[0])
        prec5_m.update(prec5, output.shape[0])

        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % cfg.LOG_INTERVAL == 0:
            lr = optimizer.get_lr()

            if local_rank == 0:
                logger.info(
                    'Train: {} [{:>4d}/{}] '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f}) '
                    'Prec@1: {top1.val:>7.4f} ({top1.avg:>7.4f}) '
                    'Prec@5: {top5.val:>7.4f} ({top5.avg:>7.4f}) '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                    'LR: {lr:.3e}'
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx,
                        len(loader),
                        loss = losses_m,
                        top1 = prec1_m,
                        top5 = prec5_m,
                        batch_time = batch_time_m,
                        rate = input.shape[0] * cfg.NUM_GPU / batch_time_m.val,
                        rate_avg = input.shape[0] * cfg.NUM_GPU / batch_time_m.avg,
                        lr = lr,
                        data_time = data_time_m))
                
                if type(writer) != type(None):
                    writer.add_scalar(
                        'Loss/train',
                        prec1_m.avg,
                        epoch * len(loader) + batch_idx)
                    writer.add_scalar(
                        'Accuracy/train',
                        prec1_m.avg,
                        epoch * len(loader) + batch_idx)
                    writer.add_scalar(
                        'Learning_Rate',
                        optimizer.param_groups[0]['lr'],
                        epoch * len(loader) + batch_idx)

                if cfg.SAVE_IMAGES and output_dir:
                    image = Image.fromarray(input.numpy())
                    file_name = os.path.join(
                            output_dir, 'train-batch-%d.jpg' %
                            batch_idx)
                    with Image.open(file_name) as f:
                        image.save(f)

        if saver is not None and cfg.RECOVERY_INTERVAL and (
                last_batch or (batch_idx + 1) % cfg.RECOVERY_INTERVAL == 0):
            saver.save_recovery(
                model,
                optimizer,
                cfg,
                epoch,
                model_ema = model_ema,
                use_amp = use_amp,
                batch_idx = batch_idx)

        end = time.time()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])
