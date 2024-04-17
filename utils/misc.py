'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging

import torch

import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter', 'ce_loss']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)  # 按topk最大值构建张量
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # topk返回两个张量：values和indices，分别对应前k大值的数值和索引
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    # eq输出元素相等的布尔值,相等返回true，不相等返回False
    # expand_as将张量扩展为pred的大小
    # reshape()重新定义矩阵的形状

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))  # 以百分比形式输出
    return res


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)  # log_softmax函数能够对softmax函数的输出进行取对数，并保持概率分布的性质。具体来说，log_softmax函数的作用是对输入的向量进行指数运算，将结果除以向量元素的和，再进行对数运算，得到一个新的向量。
        return F.nll_loss(log_pred, targets, reduction=reduction)  # F.nll_loss计算交叉熵，在函数内部不含有提前使用softmax转化的部分
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable 不稳定
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)  # * 操作可以执行逐元素矩阵乘法,类似torch.mul()
        return nll_loss


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
