'''
本文件为原lib/models/core/test.py的转写
'''

import time
from collections import OrderedDict

import paddle
from lib.utils.pimm.utils import accuracy, reduce_tensor
from lib.utils.util import AverageMeter


# validate function
def validate(
    epoch, 
    model, 
    loader, 
    loss_fn, cfg, 
    log_suffix = '', 
    logger = None, 
    writer = None, 
    local_rank = 0):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with paddle.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            output = model(input)
            
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = cfg.TTA
            if reduce_factor > 1:
                output = output.unfold(
                    0,
                    reduce_factor,
                    reduce_factor).mean(axis = 2)
                target = target[0:target.shape[0]:reduce_factor]

            loss = loss_fn(output, target)
            prec1, prec5 = accuracy(output, target, topk = (1, 5))

            if cfg.NUM_GPU > 1:
                reduced_loss = reduce_tensor(loss.data, cfg.NUM_GPU)
                prec1 = reduce_tensor(prec1, cfg.NUM_GPU)
                prec5 = reduce_tensor(prec5, cfg.NUM_GPU)
            else:
                reduced_loss = loss

            losses_m.update(reduced_loss.item(), input.shape[0])
            prec1_m.update(prec1, output.shape[0])
            prec5_m.update(prec5, output.shape[0])

            batch_time_m.update(time.time() - end)
            end = time.time()
            if local_rank == 0 and (batch_idx % 10 == 0 or last_batch):
                log_name = 'Test' + log_suffix
                logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Prec@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Prec@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, 
                        batch_idx, 
                        last_idx,
                        batch_time = batch_time_m, 
                        loss = losses_m,
                        top1 = prec1_m, 
                        top5 = prec5_m))

                if type(writer) != type(None):
                    writer.add_scalar(
                        'Loss' + log_suffix + '/vaild',
                        prec1_m.avg,
                        epoch * len(loader) + batch_idx)
                    writer.add_scalar(
                        'Accuracy' + log_suffix + '/vaild',
                        prec1_m.avg,
                        epoch * len(loader) + batch_idx)

    metrics = OrderedDict(
        [('loss', losses_m.avg), 
            ('prec1', prec1_m.avg), 
            ('prec5', prec5_m.avg)])

    return metrics
