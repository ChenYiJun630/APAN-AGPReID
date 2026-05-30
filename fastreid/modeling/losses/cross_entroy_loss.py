# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F

from fastreid.utils.events import get_event_storage


def log_accuracy(pred_class_logits, gt_classes, topk=(1,)):
    """
    Log the accuracy metrics to EventStorage.
    """
    bsz = pred_class_logits.size(0)
    maxk = max(topk)
    _, pred_class = pred_class_logits.topk(maxk, 1, True, True)
    pred_class = pred_class.t()
    correct = pred_class.eq(gt_classes.view(1, -1).expand_as(pred_class))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / bsz))

    storage = get_event_storage()
    storage.put_scalar("cls_accuracy", ret[0])


def cross_entropy_loss(pred_class_outputs, gt_classes, eps, alpha=0.2):
    num_classes = pred_class_outputs.size(1)
    # eps: 这是一个浮点数，用于控制标签平滑的程度。
    # 如果 eps >= 0，它将被用作固定的平滑参数。
    # 如果 eps < 0，将启用自适应标签平滑。
    if eps >= 0:
        smooth_param = eps
    else:
        # Adaptive label smooth regularization
        soft_label = F.softmax(pred_class_outputs, dim=1)#转换为概率分布。dim=1 表示对每个样本的类别维度进行Softmax。
        smooth_param = alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)
        # 这一步非常关键。它通过索引 gt_classes 从 soft_label 中提取出每个样本真实类别的预测概率。
        #torch.arange(soft_label.size(0)) 生成一个从0到 batch_size-1 的序列，用于选择每个样本
        #结合起来，soft_label[torch.arange(soft_label.size(0)), gt_classes] 得到了一个形状为 (batch_size,) 的张量，其中包含每个样本真实类别的预测概率

        # mooth_param = alpha * ...: 将提取出的真实类别预测概率乘以 alpha 作为自适应的平滑参数。这意味着模型对某个样本真实类别的预测越自信，其平滑参数 smooth_param 就越大，
        # 从而在损失计算中对该真实类别的惩罚相对越小（因为目标值更接近1），同时对其他类别的“惩罚”相对越大（目标值更小）。这是一种“信任当前预测”的平滑方式。。

    log_probs = F.log_softmax(pred_class_outputs, dim=1)
    #这一步计算了模型输出的对数Softmax概率。在深度学习中，通常计算 log_softmax 而不是 softmax，因为它可以与交叉熵损失结合，提供数值稳定性，并且 log(P) 的梯度比 P 更容易计算
    with torch.no_grad():
        targets = torch.ones_like(log_probs)#初始化一个与 log_probs 形状相同的张量 targets，所有元素都为1。
        targets *= smooth_param / (num_classes - 1)
       
        targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

    loss = (-targets * log_probs).sum(dim=1)


    #归一化损失
    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)
    loss = loss.sum() / non_zero_cnt

    return loss
