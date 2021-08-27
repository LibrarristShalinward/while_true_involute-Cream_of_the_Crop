from os.path import isfile
import logging
import pickle
from collections import OrderedDict

# 原timm.models.helpers.resume_checkpoint
# 将存储的文件类型改为了.bin，读取依赖为pickle
def resume_checkpoint(model, checkpoint_path):
    other_state = {}
    resume_epoch = None
    if isfile(checkpoint_path):
        with open(checkpoint_path, "br") as f:
            checkpoint = pickle.load(f)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            if 'optimizer' in checkpoint:
                other_state['optimizer'] = checkpoint['optimizer']
            if 'amp' in checkpoint:
                other_state['amp'] = checkpoint['amp']
            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1
            logging.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            logging.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return other_state, resume_epoch
    else:
        logging.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()