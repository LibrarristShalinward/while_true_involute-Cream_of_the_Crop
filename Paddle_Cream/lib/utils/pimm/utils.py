import operator
import os
import logging
from copy import deepcopy
import paddle
import glob
from collections import OrderedDict


# 原timm.utils.CheckpointSaver
class CheckpointSaver:
    def __init__(
        self,
        checkpoint_prefix='checkpoint',
        recovery_prefix='recovery',
        checkpoint_dir='',
        recovery_dir='',
        decreasing=False,
        max_history=10):

        # state
        self.checkpoint_files = []
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ''
        self.last_recovery_file = ''

        # config
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = '.pdparams'
        self.decreasing = decreasing
        self.cmp = operator.lt if decreasing else operator.gt
        self.max_history = max_history
        assert self.max_history >= 1
    
    def save_checkpoint(self, model, optimizer, args, epoch, model_ema=None, metric=None, use_amp=False):
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        self._save(tmp_save_path, model, optimizer, args, epoch, model_ema, metric, use_amp)
        if os.path.exists(last_save_path):
            os.unlink(last_save_path)
        os.rename(tmp_save_path, last_save_path)
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (len(self.checkpoint_files) < self.max_history
                or metric is None or self.cmp(metric, worst_file[1])):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            os.link(last_save_path, save_path)
            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[1],
                reverse=not self.decreasing)

            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += ' {}\n'.format(c)
            logging.info(checkpoints_str)

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                best_save_path = os.path.join(self.checkpoint_dir, 'model_best' + self.extension)
                if os.path.exists(best_save_path):
                    os.unlink(best_save_path)
                os.link(last_save_path, best_save_path)

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)
    
    def _save(self, save_path, model, optimizer, args, epoch, model_ema=None, metric=None, use_amp=False):
        save_state = {
            'epoch': epoch,
            'arch': args.model,
            'state_dict': get_state_dict(model),
            'optimizer': optimizer.state_dict(),
            'args': args,
            'version': 2,
        }
        if model_ema is not None:
            save_state['state_dict_ema'] = get_state_dict(model_ema)
        if metric is not None:
            save_state['metric'] = metric
        paddle.save(save_state, save_path)
    
    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index <= 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                logging.debug("Cleaning checkpoint: {}".format(d))
                os.remove(d[0])
            except Exception as e:
                logging.error("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]
    
    def save_recovery(self, model, optimizer, args, epoch, model_ema=None, use_amp=False, batch_idx=0):
        assert epoch >= 0
        filename = '-'.join([self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        self._save(save_path, model, optimizer, args, epoch, model_ema, use_amp=use_amp)
        if os.path.exists(self.last_recovery_file):
            try:
                logging.debug("Cleaning recovery: {}".format(self.last_recovery_file))
                os.remove(self.last_recovery_file)
            except Exception as e:
                logging.error("Exception '{}' while removing {}".format(e, self.last_recovery_file))
        self.last_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path
    
    def find_recovery(self):
        recovery_path = os.path.join(self.recovery_dir, self.recovery_prefix)
        files = glob.glob(recovery_path + '*' + self.extension)
        files = sorted(files)
        if len(files):
            return files[0]
        else:
            return ''

# 原timm.utils.ModelEma
class ModelEma:
    def __init__(self, model, decay=0.9999, device='', resume=''):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.stop_gradient = True

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = paddle.load(checkpoint_path)
        assert isinstance(checkpoint, dict)
        if 'state_dict_ema' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict_ema'].items():
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.set_state_dict(new_state_dict)
            logging.info("Loaded state_dict_ema")
        else:
            logging.warning("Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with paddle.no_grad():
            msd = model.state_dict()
            state_dict = self.ema.state_dict()
            for k, ema_v in state_dict.items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                # ema_v.clone(ema_v * self.decay + (1. - self.decay) * model_v)
                state_dict[k] = ema_v * self.decay + (1. - self.decay) * model_v
            self.ema.set_state_dict(state_dict)


# 原timm.utils.reduce_tensor
def reduce_tensor(tensor, n):
    rt = tensor.clone()
    paddle.distributed.all_reduce(rt)
    rt /= n
    return rt

# 原timm.utils.get_state_dict
def get_state_dict(model):
    return unwrap_model(model).state_dict()

# 原timm.utils.unwrap_model
def unwrap_model(model):
    if isinstance(model, ModelEma):
        return unwrap_model(model.ema)
    else:
        return model.network if hasattr(model, 'network') else model
