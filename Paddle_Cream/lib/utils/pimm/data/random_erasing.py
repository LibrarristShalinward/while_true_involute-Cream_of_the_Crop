'''
本文件为timm.data.random_erasing的全复制修改
由于引用过于复杂，故不对未被调用的函数进行筛除
'''
import math
import random

import paddle


def _get_pixels(per_pixel, rand_color, patch_size, dtype="float32"):
    if per_pixel:
        return paddle.normal(shape = patch_size)
    elif rand_color:
        return paddle.normal(shape = (patch_size[0], 1, 1))
    else:
        return paddle.zeros((patch_size[0], 1, 1))


class RandomErasing:
    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=0):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype)
                    break

    def __call__(self, input):
        if len(input.shape) == 3:
            self._erase(input, *input.shape, input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.shape
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input
