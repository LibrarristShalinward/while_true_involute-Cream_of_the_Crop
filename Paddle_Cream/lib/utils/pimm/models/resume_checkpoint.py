from os.path import isfile
import logging
from collections import OrderedDict
import paddle

# 原timm.models.helpers.resume_checkpoint
def resume_checkpoint(model, checkpoint_path):
    other_state = {}
    resume_epoch = None
    if isfile(checkpoint_path):
        checkpoint = paddle.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.set_state_dict(new_state_dict)
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
            model.set_state_dict(checkpoint)
            logging.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return other_state, resume_epoch
    else:
        logging.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()
