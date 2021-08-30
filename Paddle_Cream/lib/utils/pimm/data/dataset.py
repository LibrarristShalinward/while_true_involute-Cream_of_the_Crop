import os
import re

import paddle
from PIL import Image

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


# 原timm.data.dataset.load_class_map
def load_class_map(filename, root = ""):
    class_to_index = {}
    class_map_path = filename
    if not os.path.exists(class_map_path):
        class_map_path = os.path.join(root, filename)
        assert os.path.exists(class_map_path)
    class_map_ext = os.path.splitext(filename)[-1].lower()
    if class_map_ext == ".txt":
        with open(class_map_path) as f:
            class_to_index = {v.strip(): k for k, v in enumerate(f)}
    else:
        assert False
    return class_to_index

# 原timm.data.dataset.find_images_and_targets
def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if class_to_idx is None:
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = zip(filenames, [class_to_idx[l] for l in labels])
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx

# 原timm.data.dataset.natural_key
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

# 原timm.data.dataset.Dataset
class Dataset(paddle.io.Dataset):
    def __init__(self, root, load_bytes = False, transform = None, class_map = ""):
        class_to_index = None
        if class_map:
            class_to_index = load_class_map(class_map, root)
        images, class_to_index = find_images_and_targets(root, class_to_idx = class_to_index)
        if len(images) == 0:
                raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        
        self.root = root
        self.samples = images
        self.images = self.samples
        self.class_to_index = class_to_index
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if target is None:
            target = paddle.zeros(0)
        return image, target

    def __len__(self):
        return len(self.images)
    
    def filenames(self, indices = [], basename = False):
        if indices:
            if basename:
                return [os.path.basename(self.samples[i][0]) for i in indices]
            else:
                return [self.samples[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.samples]
            else:
                return [x[0] for x in self.samples]
