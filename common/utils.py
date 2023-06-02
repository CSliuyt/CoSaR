import errno
import os
import random
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision.datasets.utils import check_integrity
from tqdm import tqdm
from torch import optim
from torch import nn
import shutil
from json import dump
from collections import Counter


# --------------------- Fundation Set -----------------------
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    torch.cuda.empty_cache()


# --------------------------- Acc Tool -----------------------------
class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-8)


def accuracy(y_pred, y_actual, topk=(1,), return_tensor=False):
    """
    Computes the precision@k for the specified values of k in this mini-batch
    :param y_pred   : tensor, shape -> (batch_size, n_classes)
    :param y_actual : tensor, shape -> (batch_size)
    :param topk     : tuple
    :param return_tensor : bool, whether to return a tensor or a scalar
    :return:
        list, each element is a tensor with shape torch.Size([])
    """
    maxk = max(topk)
    batch_size = y_actual.size(0)

    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        if return_tensor:
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(correct_k.item() * 100.0 / batch_size)
    return res


def evaluate(dataloader, model, dev, topk=(1,)):
    """

    :param dataloader:
    :param model:
    :param dev: devices, gpu or cpu
    :param topk: [tuple]          output the top topk accuracy
    :return:     [list[float]]    topk accuracy
    """
    model.eval()
    test_loss = AverageMeter()
    test_loss.reset()
    test_accuracy = AverageMeter()
    test_accuracy.reset()

    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader, ncols=100, ascii=' >', leave=False, desc='evaluating')):
            x = sample[0].to(dev)
            y = sample[2].to(dev)
            output = model(x)['logits']
            loss = torch.nn.functional.cross_entropy(output, y.long())
            test_loss.update(loss.item(), x.size(0))
            acc = accuracy(output.softmax(dim=1), y, topk)
            test_accuracy.update(acc[0], x.size(0))
    return {'accuracy': test_accuracy.avg, 'loss': test_loss.avg}


# ----------------------------------- optimization-related -----------------------------------
def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lr_warmup(lr_list, lr_init, warmup_end_epoch=5):
    lr_list[:warmup_end_epoch] = list(np.linspace(0, lr_init, warmup_end_epoch))
    return lr_list


def lr_scheduler(lr_init, num_epochs, warmup_end_epoch=5, mode='cosine',
                 epoch_decay_start=None, epoch_decay_ratio=None, epoch_decay_interval=None):
    """

    :param lr_init：initial learning rate
    :param num_epochs: number of epochs
    :param warmup_end_epoch: number of warm up epochs
    :param mode: {cosine, linear, step}
                  cosine:
                        lr_t = 0.5 * lr_0 * (1 + cos(t * pi / T)) in t'th epoch of T epochs
                  linear:
                        lr_t = (T - t) / (T - t_decay) * lr_0, after t_decay'th epoch
                  step:
                        lr_t = lr_0 * ratio**(t//interval), e.g. ratio = 0.1 with interval = 30;
                                                                 ratio = 0.94 with interval = 2
    :param epoch_decay_start: used in linear mode as `t_decay`
    :param epoch_decay_ratio: used in step mode as `ratio`
    :param epoch_decay_interval: used in step mode as `interval`
    :return:
    """
    lr_list = [lr_init] * num_epochs

    print('| Learning rate warms up for {} epochs'.format(warmup_end_epoch))
    lr_list = lr_warmup(lr_list, lr_init, warmup_end_epoch)

    print('| Learning rate decays in {} mode'.format(mode))
    if mode == 'cosine':
        for t in range(warmup_end_epoch, num_epochs):
            lr_list[t] = 0.5 * lr_init * (1 + math.cos((t - warmup_end_epoch + 1) * math.pi /
                                                       (num_epochs - warmup_end_epoch + 1)))
    elif mode == 'linear':
        if type(epoch_decay_start) == int and epoch_decay_start > warmup_end_epoch:
            for t in range(epoch_decay_start, num_epochs):
                lr_list[t] = float(num_epochs - t) / (num_epochs - epoch_decay_start) * lr_init
        else:
            raise AssertionError('Please specify epoch_decay_start, '
                                 'and epoch_decay_start need to be larger than warmup_end_epoch')
    elif mode == 'step':
        if type(epoch_decay_ratio) == float and type(epoch_decay_interval) == int and epoch_decay_interval < num_epochs:
            for t in range(warmup_end_epoch, num_epochs):
                lr_list[t] = lr_init * epoch_decay_ratio ** ((t - warmup_end_epoch + 1) // epoch_decay_interval)

    return lr_list


def get_smoothed_label_distribution(labels, nc, epsilon):
    smoothed_label = torch.full(size=(labels.size(0), nc), fill_value=epsilon / (nc - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1-epsilon)
    return smoothed_label.to(labels.device)


# -------------------------------- train-related ------------------------------------
def build_sgd_optimizer(params, lr, weight_decay, nesterov=True):
    return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=nesterov)


def build_adam_optimizer(params, lr):
    return optim.Adam(params, lr=lr, betas=(0.9, 0.999))


def build_cosine_lr_scheduler(optimizer, total_epochs):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)


# ------------------------------- model-related -----------------------------------
def init_weights(module, init_method='Xavier'):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'He':
                nn.init.kaiming_normal_(m.weight.data)
            elif init_method == 'Xavier':
                nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, val=0)


def frozen_layer(module):
    for parameters in module.parameters():
        parameters.required_grad = False


# ------------------ Save Tool -------------------
def save_config(params, params_file):
    config_file_path = params.cfg_file
    shutil.copy(config_file_path, params_file)


def save_params(params, params_file, json_format=False):
    with open(params_file, 'w') as f:
        if not json_format:
            params_file.replace('.json', '.txt')
            for k, v in params.__dict__.items():
                f.write(f'{k:<20}: {v}\n')
        else:
            params_file.replace('.txt', '.json')
            dump(params.__dict__, f, indent=4)


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    print('save successful!')


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def str_is_int(x):
    if x.count('-') > 1:
        return False
    if x.isnumeric():
        return True
    if x.startswith('-') and x.replace('-', '').isnumeric():
        return True
    return False


def str_is_float(x):
    if str_is_int(x):
        return False
    try:
        _ = float(x)
        return True
    except ValueError:
        return False


class Config(object):
    def set_item(self, key, value):
        if isinstance(value, str):
            if str_is_int(value):
                value = int(value)
            elif str_is_float(value):
                value = float(value)
            elif value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'none':
                value = None
        if key.endswith('milestones'):
            try:
                tmp_v = value[1:-1].split(',')
                value = list(map(int, tmp_v))
            except:
                raise AssertionError(f'{key} is: {value}, format not supported!')
        self.__dict__[key] = value

    def __repr__(self):
        ret = 'Config:\n{\n'
        for k in self.__dict__.keys():
            s = f'    {k}: {self.__dict__[k]}\n'
            ret += s
        ret += '}\n'
        return ret


def load_from_cfg(path):
    # try easydict
    cfg = Config()
    if not path.endswith('.cfg'):
        path = path + '.cfg'
    if not os.path.exists(path) and os.path.exists('config' + os.sep + path):
        path = 'config' + os.sep + path
    assert os.path.isfile(path), f'{path} is not a valid config file.'

    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    for line in lines:
        if line.startswith('['):
            continue
        k, v = line.replace(' ', '').split('=')
        cfg.set_item(key=k, value=v)
    cfg.set_item(key='cfg_file', value=path)

    return cfg


def get_counter(num_classes, device):
    collect = {}
    for i in range(num_classes):
        collect[i] = []
    return collect


def get_avg_counter(num_classes, device):
    collect = {}
    for i in range(num_classes):
        collect[i] = 1.
    return collect


def to_tensor(collect, device):
    for i in range(len(collect)):
        collect[i] = torch.tensor(collect[i]).to(device)
    return collect


if __name__ == '__main__':
    import torch
    z = torch.zeros(100)
    x = torch.tensor([1, 2, 3, 1], dtype=torch.long)
    label = get_smoothed_label_distribution(x, z.size(0), (1-1/z.size(0)))
    print(label)
    a = get_counter(10)
    print(a)