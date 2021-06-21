import os
import shutil

import matplotlib
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from clr import CyclicLR


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num = 2, alpha=None, gamma = 3, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim = 1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def correct(output, target, topk=(1,)):
    """Computes the correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0).item()
        res.append(correct_k)
    return res

def save_checkpoint(state, is_best, filepath='./', filename='checkpoint.pth.tar'):
    save_path = os.path.join(filepath, filename)
    best_path = os.path.join(filepath, 'model_best.pth.tar')
    torch.save(state, save_path, _use_new_zipfile_serialization=False)
    if is_best:
        shutil.copyfile(save_path, best_path)

def find_bounds_clr(model, loader, optimizer, criterion, device, dtype, min_lr=8e-6, max_lr=8e-5, step_size=2000,
                    mode='triangular', save_path='.'):
    model.train()
    correct1, correct2 = 0, 0
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size=step_size, mode=mode)
    epoch_count = step_size // len(loader)  # Assuming step_size is multiple of batch per epoch
    accuracy = []
    for _ in trange(epoch_count):
        for batch_idx, (data, target) in enumerate(tqdm(loader)):
            if scheduler is not None:
                scheduler.batch_step()
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            corr = correct(output, target)
            accuracy.append(corr[0] / data.shape[0])

    lrs = np.linspace(min_lr, max_lr, step_size)
    plt.plot(lrs, accuracy)
    plt.show()
    plt.savefig(os.path.join(save_path, 'find_bounds_clr.png'))
    np.save(os.path.join(save_path, 'acc.npy'), accuracy)
    return
