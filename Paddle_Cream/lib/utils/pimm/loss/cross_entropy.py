import warnings
import paddle
from paddle.framework import dtype
from paddle.nn import Layer, functional
import paddle

# 原timm.loss.LabelSmoothingCrossEntropy
class LabelSmoothingCrossEntropy(Layer):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = functional.log_softmax(x, axis = -1)
        nll_loss = -logprobs.gather(axis = logprobs.dim() - 1, index = target.unsqueeze(1))
        # nll_loss = -logprobs.gather(axis=-1, index=target.unsqueeze(1))
        nll_loss = paddle.diag(nll_loss.reshape((target.shape[0], target.shape[0])))
        # nll_loss = []
        # warnings.warn(str(logprobs.shape))
        # for i in range(target.shape[0]):
        #     nll_loss.append(-logprobs[i][target[i]])
        # nll_loss = paddle.to_tensor(nll_loss)
        # warnings.warn(str(nll_loss[:5]))
        smooth_loss = -logprobs.mean(axis=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
