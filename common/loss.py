import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from common.utils import get_smoothed_label_distribution
import numpy as np


def entropy(p):
    return Categorical(probs=p).entropy()


def entropy_loss(logits, reduction='mean'):
    losses = entropy(F.softmax(logits, dim=1))  # (N)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def cross_entropy(logits, labels, reduction='mean'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    if len(labels.size()) < 2:
        labels = F.one_hot(labels, logits.size(-1))
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(
        1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def contrast_cross_entropy(logits, labels, tau=0.5, reduction='none', eps=1e-8):
    if len(labels.size()) < 2:
        labels = F.one_hot(labels.long(), logits.size(1))
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(
        1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    with torch.no_grad():
        log_logits = 1 / (F.log_softmax(logits, dim=1) * tau + eps)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


def kl_div(p, q, base=2):
    # p, q is in shape (batch_size, n_classes)
    if base == 2:
        return (p * p.log2() - p * q.log2()).sum(dim=1)
    else:
        return (p * p.log() - p * q.log()).sum(dim=1)


def js_div(p, q, base=2):
    # Jensen-Shannon divergence, value is in (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, base) + 0.5 * kl_div(q, m, base)


def loss_Jo_AR(prob_weak, prob_strong, target, cls_js_avg, cls_js_list, cls_js_list2, net_prob):
    with torch.no_grad():
        smoothed_target = get_smoothed_label_distribution(target, prob_weak.size(1), 0.1)
        noise_target = (smoothed_target + net_prob.softmax(dim=1)) / 2
        sharpen_noise_target = (noise_target / 0.1).softmax(dim=1)
        prob_target = torch.argmax(prob_weak.softmax(dim=1), dim=1)
        similar_target = get_smoothed_label_distribution(target, prob_weak.size(1), cls_js_avg)

    clean_loss = cross_entropy(prob_weak, target, reduction='none')
    noise_loss = cross_entropy(prob_strong, sharpen_noise_target, reduction='none')
    ar_loss = contrast_cross_entropy(prob_weak, target, reduction='none')
    similar_loss = cross_entropy(prob_weak, similar_target, reduction='none')

    if_clean = cls_js_list < cls_js_avg
    if_noise = cls_js_list >= cls_js_avg

    non_consistency_idx = if_noise * (prob_target != target)
    consistency_idx = if_noise * (prob_target == target)

    clean_loss = (1 - cls_js_list) * if_clean * clean_loss

    similar_loss = cls_js_avg * consistency_idx * similar_loss

    noise_loss = cls_js_list * non_consistency_idx * noise_loss

    ar_loss = cls_js_list * non_consistency_idx * ar_loss

    loss = (torch.sum(clean_loss) + torch.sum(similar_loss) + torch.sum(noise_loss) + torch.sum(
        ar_loss)) / prob_weak.size(0)
    if float('nan') in loss:
        print(clean_loss, similar_loss, noise_loss, ar_loss)
    cls_js_list2[non_consistency_idx] = js_div(prob_strong.softmax(dim=1), sharpen_noise_target)[non_consistency_idx]
    if sum(if_clean) == 0:
        return loss, cls_js_list2, ~non_consistency_idx
    return loss, cls_js_list2, if_clean


def get_js_list(prob, target, eps=1e-8):
    target = get_smoothed_label_distribution(target, prob.size(1), 0.1) + eps
    return js_div(prob + eps, target)


def get_js_avg(js_list, Theta=0.2, threshld=0.7):
    js_avg = js_list.mean()

    return js_avg


if __name__ == '__main__':
    label = torch.arange(10)
    label = get_smoothed_label_distribution(label, 10, 0.01)
    print(label)
