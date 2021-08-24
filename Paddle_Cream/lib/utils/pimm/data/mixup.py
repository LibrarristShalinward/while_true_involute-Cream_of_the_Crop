'''
本文件为timm.data.random_erasing的全复制
删除了不会用到的mixup_batch()函数
'''
import numpy as np
import paddle


def one_hot(x, num_classes, on_value=1., off_value=0.):
    x = x.long().view(-1, 1)
    return paddle.full((x.size()[0], num_classes), off_value).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value)
    return lam*y1 + (1. - lam)*y2


class FastCollateMixup:

    def __init__(self, mixup_alpha=1., label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mixup_enabled = True

    def __call__(self, batch):
        batch_size = len(batch)
        lam = 1.
        if self.mixup_enabled:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        target = paddle.to_tensor([b[1] for b in batch], dtype="int64")
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, device='cpu')

        tensor = paddle.zeros((batch_size, *batch[0][0].shape), dtype="uint8")
        for i in range(batch_size):
            mixed = batch[i][0].astype(np.float32) * lam + \
                    batch[batch_size - i - 1][0].astype(np.float32) * (1 - lam)
            np.round(mixed, out=mixed)
            tensor[i] += paddle.to_tensor(mixed.astype(np.uint8))

        return tensor, target
