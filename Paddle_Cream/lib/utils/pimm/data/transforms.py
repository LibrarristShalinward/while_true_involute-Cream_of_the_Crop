import re
from PIL import Image
from paddle.framework import dtype
from paddle.vision.transforms import crop, resize, RandomHorizontalFlip, ColorJitter, Normalize
import warnings
import random
import math
from .auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
import numpy as np
import paddle

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# 原timm.data.transforms变量
_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


# 原timm.data.transforms_factory.create_transform
def create_transform(
        input_size,
        is_training=False,
        use_prefetcher=False,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=None,
        tf_preprocessing=False,
        separate=False):

    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    
    if is_training:
        transform = transforms_imagenet_train(
            img_size,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
            separate=separate)
    else:
        assert not separate, "Separate transforms not supported for validation preprocessing"
        transform = transforms_imagenet_eval(
            img_size,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
            crop_pct=crop_pct)

    return transform

# 原timm.data.transforms_factory.transforms_imagenet_train
def transforms_imagenet_train(
        img_size=224,
        scale=(0.08, 1.0),
        color_jitter=0.4,
        auto_augment=None,
        interpolation='random',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        separate=False,
):
    primary_tfl = [
        RandomResizedCropAndInterpolation(
            img_size, scale=scale, interpolation=interpolation),
        RandomHorizontalFlip()
    ]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = _pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]
    elif color_jitter is not None:
        if isinstance(color_jitter, (list, tuple)):
            assert len(color_jitter) in (3, 4)
        else:
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [ColorJitter(*color_jitter)]

    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            ToTensor(),
            Normalize(
                mean=mean,
                std=std)
        ]
        if re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu'))

    if separate:
        return transforms.Compose(primary_tfl), transforms.Compose(secondary_tfl), transforms.Compose(final_tfl)
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)

# 原timm.data.transforms.RandomResizedCropAndInterpolation
class RandomResizedCropAndInterpolation:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return resized_crop(img, i, j, h, w, self.size, interpolation)

# 原timm.data.transforms._pil_interp
def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR

# 原timm.data.transforms.ToNumpy
class ToNumpy:
    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img

# 原timm.data.transforms.ToTensor
class ToTensor:

    def __init__(self, dtype="float32"):
        self.dtype = dtype

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)
        return paddle.to_tensor(np_img, dtype = dtype)

# 原torchvision.transforms.functional.resized_crop
def resized_crop(img, i, j, h, w, size, interpolation=Image.BILINEAR):
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img