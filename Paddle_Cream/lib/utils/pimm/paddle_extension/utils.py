'''
本文件用于放置Cream中用到而paddle中没有对应实现的torch-API的补充实现
'''
from paddle.io import DistributedBatchSampler, DataLoader
from paddle.nn import Layer
from paddle import Tensor


# 对标torch.utils.data.distributed.DistributedSampler
# 官方补充
# 来源：https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/pytorch_project_convertor/API_docs/utils/torch.utils.data.distributed.DistributedSampler.md
class DistributedSampler(DistributedBatchSampler):
    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0,
                 drop_last=False):
        super().__init__(
            dataset=dataset,
            batch_size=1,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last)

# 对标带有sampler属性的torch.utils.data.DataLoader
# 官方补充
# 来源：https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/pytorch_project_convertor/API_docs/utils/torch.utils.data.DataLoader.md
# 修改了最后一行错误：self-seld
class DataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sampler=None,
                 num_workers=0,
                 collate_fn=None,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None,):
        if isinstance(dataset[0], (tuple, list)):
            return_list = True
        else:
            return_list = False
        return_list = True
        super().__init__(
            dataset,
            feed_list=None,
            places=None,
            return_list=return_list,
            batch_sampler=batch_sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=collate_fn,
            num_workers=num_workers,
            use_buffer_reader=True,
            use_shared_memory=False,
            timeout=timeout,
            worker_init_fn=worker_init_fn)
        if sampler is not None:
            self.batch_sampler.sampler = sampler

# 对标torch.nn.Identity
# 参考来源：https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Identity
class Identity(Layer):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input