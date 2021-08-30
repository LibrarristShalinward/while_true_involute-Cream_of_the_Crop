import numpy as np
import paddle
from paddle.framework import dtype

from ..paddle_extension import DataLoader, DistributedSampler
from .mixup import FastCollateMixup
from .random_erasing import RandomErasing
from .transforms import create_transform

# 原timm.data.constants变量
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# 原timm.data.loader.create_loader
def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        collate_fn=None,
        fp16=False,
        tf_preprocessing=False,
        no_cuda = True
):
    re_num_splits = 0
    if re_split:
        re_num_splits = num_aug_splits or 2
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0,
    )

    sampler = None
    if distributed:
        if is_training:
            sampler = DistributedSampler(dataset)
        else:
            assert False, "无法在预测模型下启动分布式计算"

    if collate_fn is None and use_prefetcher:
        collate_fn = fast_collate

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=is_training,
    )
    if use_prefetcher:
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            fp16=fp16,
            re_prob=re_prob if is_training else 0.,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits, 
            no_cuda = no_cuda
        )

    return loader

# 原timm.data.loader.PrefetchLoader
class PrefetchLoader:

    def __init__(self,
                 loader,
                 mean=IMAGENET_DEFAULT_MEAN,
                 std=IMAGENET_DEFAULT_STD,
                 fp16=False,
                 re_prob=0.,
                 re_mode='const',
                 re_count=1,
                 re_num_splits=0, 
                 no_cuda = True):
        self.loader = loader
        if no_cuda:
            self.mean = paddle.to_tensor([x * 255 for x in mean]).reshape((1, 3, 1, 1))
            self.std = paddle.to_tensor([x * 255 for x in std]).reshape((1, 3, 1, 1))
        else:
            self.mean = paddle.to_tensor([x * 255 for x in mean]).cuda().reshape((1, 3, 1, 1))
            self.std = paddle.to_tensor([x * 255 for x in std]).cuda().reshape((1, 3, 1, 1))
        
        self.fp16 = fp16
        if fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        if re_prob > 0.:
            self.random_erasing = RandomErasing(
                probability=re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits)
        else:
            self.random_erasing = None
        
        self.no_cuda = no_cuda

    def __iter__(self):
        first = True

        for next_input, next_target in self.loader:
            if not self.no_cuda:
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
            if self.fp16:
                next_input = (paddle.to_tensor(next_input, dtype = "float16") - self.mean) / (self.std)
            else:
                next_input = (paddle.to_tensor(next_input, dtype = "float32") - self.mean) / (self.std)
            if self.random_erasing is not None:
                next_input = self.random_erasing(next_input)

            if not first:
                yield input, target
            else:
                first = False

            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def mixup_enabled(self):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            return self.loader.collate_fn.mixup_enabled
        else:
            return False

    @mixup_enabled.setter
    def mixup_enabled(self, x):
        if isinstance(self.loader.collate_fn, FastCollateMixup):
            self.loader.collate_fn.mixup_enabled = x

# 原timm.data.loader.fast_collate
def fast_collate(batch):
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one tensor ordered by position
        # such that all tuple of position n will end up in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = paddle.zeros(flattened_batch_size, dtype="int64")
        tensor = paddle.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype="uint8")
        for i in range(batch_size):
            assert len(batch[i][0]) == inner_tuple_size  # all input tensor tuples must be same length
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += paddle.to_tensor(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = paddle.to_tensor([b[1] for b in batch], dtype="int64")
        assert len(targets) == batch_size
        tensor = paddle.zeros((batch_size, *batch[0][0].shape), dtype="int64")
        for i in range(batch_size):
            tensor[i] += paddle.to_tensor(batch[i][0])
        tensor = paddle.to_tensor(tensor, dtype = "uint8")
        return tensor, targets
    elif isinstance(batch[0][0], paddle.Tensor):
        targets = paddle.to_tensor([b[1] for b in batch], dtype="int64")
        assert len(targets) == batch_size
        tensor = paddle.to_tensor((batch_size, *batch[0][0].shape), dtype="uint8")
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False


